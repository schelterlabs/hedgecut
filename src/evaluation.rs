use crate::dataset::{Sample, Dataset};
use crate::tree::ExtremelyRandomizedTrees;
use rand::{thread_rng, RngCore};

pub fn evaluate<S: Sample + Sync>(
    name: &str,
    trees: &ExtremelyRandomizedTrees,
    test_data: &Vec<S>
) {
    let mut t_p = 0;
    let mut _f_p = 0;
    let mut t_n = 0;
    let mut _f_n = 0;

    for sample in test_data.iter() {
        let predicted_label = trees.predict(sample);

        if sample.true_label() {
            if predicted_label {
                t_p += 1;
            } else {
                _f_n += 1;
            }
        } else {
            if predicted_label {
                _f_p += 1;
            } else {
                t_n += 1;
            }
        }
    }

    let accuracy = (t_p + t_n) as f64 / test_data.len() as f64;
    //println!("Accuracy {}", accuracy);
    //println!("[{}\t{}\n{}\t{}]", t_p, f_n, f_p, t_n);
    println!("{},hedgecut,{}", name, accuracy);
}

pub fn end_to_end<D: Dataset + Sync, S: Sample + Sync>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    test_data: Vec<S>,
    _seed: u64,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split: usize,
) {

    let mut rng = thread_rng();
    let seed = rng.next_u64();

    let trees = ExtremelyRandomizedTrees::fit(
        &dataset,
        samples,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );

    //let training_duration = training_start.elapsed();
    //println!("Fitted {} trees in {} ms", num_trees, training_duration.as_millis());

    evaluate(name, &trees, &test_data);

    //let removal1_start = Instant::now();
    //trees.forget(&sample_to_forget);
    //let removal1_duration = removal1_start.elapsed();
    //println!("Removed sample in {} μs", removal1_duration.as_micros());

    //evaluate(&trees, &test_data);

    //let removal2_start = Instant::now();
    //trees.forget(&another_sample_to_forget);
    //let removal2_duration = removal2_start.elapsed();
    //println!("Removed sample in {} μs", removal2_duration.as_micros());
    //evaluate(&trees, &test_data);
}