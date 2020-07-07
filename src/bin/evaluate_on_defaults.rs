extern crate hedgecut;

use std::time::Instant;

use hedgecut::tree::ExtremelyRandomizedTrees;
use hedgecut::dataset::DefaultsDataset;
use hedgecut::evaluation::evaluate;

fn main() {

    let samples = DefaultsDataset::samples_from_csv("datasets/defaults-train.csv");
    let test_data = DefaultsDataset::samples_from_csv("datasets/defaults-test.csv");

    let sample_to_forget = samples.get(0).unwrap().clone();
    let another_sample_to_forget = samples.get(20).unwrap().clone();

    let dataset = DefaultsDataset::from_samples(&samples);

    let seed: u64 = 4545;
    let num_trees = 100;
    let min_leaf_size = 2;
    let max_tries_per_split = 50;

    let training_start = Instant::now();

    let mut trees = ExtremelyRandomizedTrees::fit(
        &dataset,
        samples,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );

    let training_duration = training_start.elapsed();
    println!("Fitted {} trees in {} ms", num_trees, training_duration.as_millis());

    evaluate(&trees, &test_data);

    trees.forget(&sample_to_forget);

    evaluate(&trees, &test_data);

    trees.forget(&another_sample_to_forget);

    evaluate(&trees, &test_data);
}

