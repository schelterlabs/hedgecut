extern crate hedgecut;

use std::time::Instant;

use hedgecut::tree::ExtremelyRandomizedTrees;
use hedgecut::dataset::DefaultsDataset;

fn main() {

    let samples = DefaultsDataset::samples_from_csv("datasets/defaults-train.csv");
    let test_data = DefaultsDataset::samples_from_csv("datasets/defaults-test.csv");

    let sample_to_forget = samples.get(0).unwrap().clone();

    let dataset = DefaultsDataset::from_samples(&samples);

    let seed: u64 = 666;
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

    let mut t_p = 0;
    let mut f_p = 0;
    let mut t_n = 0;
    let mut f_n = 0;

    for sample in test_data.iter() {
        let predicted_label = trees.predict(sample);

        if sample.label {
            if predicted_label {
                t_p += 1;
            } else {
                f_n += 1;
            }
        } else {
            if predicted_label {
                f_p += 1;
            } else {
                t_n += 1;
            }
        }
    }

    let accuracy = (t_p + t_n) as f64 / test_data.len() as f64;
    println!("Accuracy {}", accuracy);
    println!("[{}\t{}\n{}\t{}]", t_p, f_n, f_p, t_n);


    trees.forget(&sample_to_forget);
}
