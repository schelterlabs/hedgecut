extern crate hedgecut;

use hedgecut::dataset::PropublicaDataset;
use hedgecut::evaluation::end_to_end;

fn main() {

    let samples = PropublicaDataset::samples_from_csv("datasets/propublica-train.csv");
    let test_data = PropublicaDataset::samples_from_csv("datasets/propublica-test.csv");
    let dataset = PropublicaDataset::from_samples(&samples);

    let seed: u64 = 4545;
    let num_trees = 100;
    let min_leaf_size = 2;
    let max_tries_per_split = 5;

    end_to_end(
        "propublica",
        dataset,
        samples,
        test_data,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );
}

