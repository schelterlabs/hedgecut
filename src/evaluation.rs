use crate::dataset::Sample;
use crate::tree::ExtremelyRandomizedTrees;

pub fn evaluate<S: Sample + Sync>(trees: &ExtremelyRandomizedTrees, test_data: &Vec<S>) {
    let mut t_p = 0;
    let mut f_p = 0;
    let mut t_n = 0;
    let mut f_n = 0;

    for sample in test_data.iter() {
        let predicted_label = trees.predict(sample);

        if sample.true_label() {
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
}