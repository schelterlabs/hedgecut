extern crate hedgecut;

use hedgecut::dataset::TitanicDataset;
use hedgecut::evaluation::end_to_end;

fn main() {

    let samples = TitanicDataset::samples_from_csv("datasets/titanic-train.csv");
    let test_data = TitanicDataset::samples_from_csv("datasets/titanic-test.csv");

    let dataset = TitanicDataset::from_samples(&samples);

    let seed: u64 = 42;
    let num_trees = 100;
    let min_leaf_size = 2;
    let max_tries_per_split = 25;

    end_to_end(
        "titanic",
        dataset,
        samples,
        test_data,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );
}
