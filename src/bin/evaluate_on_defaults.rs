extern crate hedgecut;

use std::time::Instant;

use hedgecut::tree::ExtremelyRandomizedTrees;
use hedgecut::dataset::DefaultsDataset;

fn main() {

    let train_data = DefaultsDataset::dataset_from_csv();
    let test_data = DefaultsDataset::samples_from_csv();

    let num_trees = 100;
    let max_tries_per_split = 25;

    let training_start = Instant::now();

    let trees = ExtremelyRandomizedTrees::fit(&train_data, num_trees, max_tries_per_split);

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
    println!("[{}\t{}\n{}\t{}]", t_p, f_p, f_n, t_n);
}
