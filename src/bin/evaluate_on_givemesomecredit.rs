extern crate hedgecut;

use hedgecut::dataset::GiveMeSomeCreditDataset;
use hedgecut::evaluation::end_to_end;

fn main() {

    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let test_data = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-test.csv");
    let dataset = GiveMeSomeCreditDataset::from_samples(&samples);

    let seed: u64 = 4545;
    let num_trees = 100;
    let min_leaf_size = 2;
    let max_tries_per_split = 50;

    end_to_end(
        "givemesomecredit",
        dataset,
        samples,
        test_data,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );
}

