extern crate csv;
extern crate rand;
extern crate rayon;

mod split_stats;
mod dataset;
mod tree;

use tree::ExtremelyRandomizedTrees;
use dataset::TitanicDataset;




fn main() {

    let train_data = TitanicDataset::dataset_from_csv();
    let test_data = TitanicDataset::samples_from_csv();

    let trees = ExtremelyRandomizedTrees::fit(&train_data, 100);

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
