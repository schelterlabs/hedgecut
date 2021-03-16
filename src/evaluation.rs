use crate::dataset::{Sample, Dataset};
use crate::tree::ExtremelyRandomizedTrees;
use rand::{thread_rng, RngCore, Rng};
use std::time::Instant;
use rand::seq::SliceRandom;

pub fn evaluate<S: Sample + Sync>(
    name: &str,
    trees: &ExtremelyRandomizedTrees,
    test_data: &Vec<S>,
    training_time_and_max_tries: Option<(u128, usize)>,

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

    match training_time_and_max_tries {
        Some((duration, tries)) => {
            println!("{},hedgecut,{},{},{}", name, accuracy, duration, tries)
        },
        None => println!("{},hedgecut,{}", name, accuracy),
    }

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

    let training_start = Instant::now();
    let trees = ExtremelyRandomizedTrees::fit(
        &dataset,
        samples,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );

    let training_duration = training_start.elapsed();
    println!("Fitted {} trees in {} ms", num_trees, training_duration.as_millis());

    evaluate(name, &trees, &test_data, None);

}

pub fn accuracy_forget<D: Dataset + Sync, S: Sample + Sync + Eq>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    test_data: Vec<S>,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split: usize,
) {

    let mut rng = thread_rng();
    let seed = rng.next_u64();

    let target_robustness = ((dataset.num_records() as f64) / 1000.0).round() as usize;

    let samples_to_forget: Vec<S> = (0..target_robustness)
        .map(|_| {
            let index = rng.gen_range(0, dataset.num_records());
            samples.get(index as usize).unwrap().clone()
        })
        .collect();

    let samples_for_retraining: Vec<S> = samples.iter()
        .filter_map(|s| {
            if samples_to_forget.contains(&s) {
                None
            } else {
                Some(s.clone())
            }
        })
        .collect();

    let mut trees = ExtremelyRandomizedTrees::fit(
        &dataset,
        samples,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );

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

    for sample in &samples_to_forget {
        trees.forget(sample);
    }

    let mut t_p_forget = 0;
    let mut f_p_forget = 0;
    let mut t_n_forget = 0;
    let mut f_n_forget = 0;

    for sample in test_data.iter() {
        let predicted_label = trees.predict(sample);

        if sample.true_label() {
            if predicted_label {
                t_p_forget += 1;
            } else {
                f_n_forget += 1;
            }
        } else {
            if predicted_label {
                f_p_forget += 1;
            } else {
                t_n_forget += 1;
            }
        }
    }

    let accuracy_forget = (t_p_forget + t_n_forget) as f64 / test_data.len() as f64;

    let retrained_trees = ExtremelyRandomizedTrees::fit(
        &dataset,
        samples_for_retraining,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );

    let mut t_p_retrained = 0;
    let mut f_p_retrained = 0;
    let mut t_n_retrained = 0;
    let mut f_n_retrained = 0;

    for sample in test_data.iter() {
        let predicted_label = retrained_trees.predict(sample);

        if sample.true_label() {
            if predicted_label {
                t_p_retrained += 1;
            } else {
                f_n_retrained += 1;
            }
        } else {
            if predicted_label {
                f_p_retrained += 1;
            } else {
                t_n_retrained += 1;
            }
        }
    }

    let accuracy_retrained = (t_p_retrained + t_n_retrained) as f64 / test_data.len() as f64;

    println!(
        "{},hedgecut_forget,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        name,
        target_robustness,
        accuracy,
        t_p,
        f_n,
        f_p,
        t_n,
        accuracy_forget,
        t_p_forget,
        f_n_forget,
        f_p_forget,
        t_n_forget,
        accuracy_retrained,
        t_p_retrained,
        f_n_retrained,
        f_p_retrained,
        t_n_retrained
    );
}
pub fn forget2<D: Dataset + Sync, S: Sample + Sync>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split: usize,
) {

    let mut rng = thread_rng();
    let seed = rng.next_u64();

    let target_robustness = ((dataset.num_records() as f64) / 1000.0).round() as usize;

    let samples_to_forget: Vec<S> = (0..target_robustness)
        .map(|_| {
            let index = rng.gen_range(0, dataset.num_records());
            samples.get(index as usize).unwrap().clone()
        })
        .collect();

    let mut ert = ExtremelyRandomizedTrees::fit(
        &dataset,
        samples,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );

    // let training_duration = training_start.elapsed();
    // println!("Fitted {} trees in {} ms", num_trees, training_duration.as_millis());
    use crate::tree::Tree;

    let mut total_hit = 0;
    let mut total_changed = 0;

    for sample in &samples_to_forget {
        ert.trees.iter_mut().for_each(|tree| {
            let (hit, changed) = Tree::forget_from2(tree, sample, 1);
            total_hit += hit;
            total_changed += changed;
        });

        //println!("{},hedgecut,{}", name, removal_duration.as_micros());
    }

    println!("{},{},{},{}", name, min_leaf_size, total_hit, total_changed);
}

pub fn forget<D: Dataset + Sync, S: Sample + Sync>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split: usize,
) {

    let mut rng = thread_rng();
    let seed = rng.next_u64();

    let target_robustness = ((dataset.num_records() as f64) / 1000.0).round() as usize;

    let samples_to_forget: Vec<S> = (0..target_robustness)
        .map(|_| {
            let index = rng.gen_range(0, dataset.num_records());
            samples.get(index as usize).unwrap().clone()
        })
        .collect();

    let mut trees = ExtremelyRandomizedTrees::fit(
        &dataset,
        samples,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );

    // let training_duration = training_start.elapsed();
    // println!("Fitted {} trees in {} ms", num_trees, training_duration.as_millis());

    for sample in &samples_to_forget {
        let removal_start = Instant::now();
        trees.forget(sample);
        let removal_duration = removal_start.elapsed();
        println!("{},hedgecut,{}", name, removal_duration.as_micros());
    }
}

pub fn max_tries<D: Dataset + Sync, S: Sample + Sync>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    test_data: Vec<S>,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split_candidates: Vec<usize>,
) {

    let mut rng = thread_rng();

    for _ in 0..6 {
        for max_tries_per_split in &max_tries_per_split_candidates {
            let seed = rng.next_u64();

            let training_samples = samples.clone();

            let training_start = Instant::now();
            let trees = ExtremelyRandomizedTrees::fit(
                &dataset,
                training_samples,
                seed,
                num_trees,
                min_leaf_size,
                max_tries_per_split.clone()
            );
            let training_duration = training_start.elapsed();
            evaluate(
                name,
                &trees,
                &test_data,
                Some((training_duration.as_millis(), max_tries_per_split.clone()))
            );
        }
    }

}

pub fn train_time<D: Dataset + Sync, S: Sample + Sync>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split: usize,
) {

    let mut rng = thread_rng();

    let seed = rng.next_u64();

    let training_samples = samples.clone();

    let training_start = Instant::now();
    ExtremelyRandomizedTrees::fit(
        &dataset,
        training_samples,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split.clone()
    );
    let training_duration = training_start.elapsed();
    println!("{},hedgecut,{}", name, training_duration.as_millis());
}

pub fn robustness<D: Dataset + Sync, S: Sample + Sync>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split: usize,
) {

    let mut rng = thread_rng();

    let seed = rng.next_u64();

    for epsilon_factor in &[10000, 5000, 1000, 500, 100, 50] {
        let training_samples = samples.clone();
        let epsilon = 1.0 / *epsilon_factor as f64;

        let training_start = Instant::now();
        ExtremelyRandomizedTrees::fit_with_epsilon(
            &dataset,
            training_samples,
            seed,
            num_trees,
            min_leaf_size,
            max_tries_per_split.clone(),
            epsilon
        );
        let training_duration = training_start.elapsed();
        println!("{},hedgecut,{},{}", name, epsilon_factor, training_duration.as_millis());
    }
}

pub fn robustness2<D: Dataset + Sync, S: Sample + Sync>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split: usize,
) {

    let mut rng = thread_rng();

    let seed = rng.next_u64();

    for epsilon_factor in &[10000, 5000, 1000, 500, 100, 50] {
        let training_samples = samples.clone();
        let epsilon = 1.0 / *epsilon_factor as f64;

        let ert = ExtremelyRandomizedTrees::fit_with_epsilon(
            &dataset,
            training_samples,
            seed,
            num_trees,
            min_leaf_size,
            max_tries_per_split.clone(),
            epsilon
        );

        let mut num_robust = 0;
        let mut num_non_robust = 0;

        for tree in ert.trees {
            let (tree_num_robust, tree_num_non_robust) = node_count(tree);
            num_robust += tree_num_robust;
            num_non_robust += tree_num_non_robust;
        }
        let ratio = num_non_robust as f64 / (num_non_robust + num_robust) as f64;

        println!("{},hedgecut,{},{},{},{}", name, epsilon_factor, ratio, num_robust, num_non_robust);
    }
}

pub fn node_count(tree: crate::tree::Tree) -> (usize, usize) {

    let mut num_robust = 0;
    let mut num_non_robust = 0;

    num_robust += tree.num_robust_nodes;
    num_non_robust += tree.num_non_robust_nodes;

    for (_, subtrees) in tree.alternative_subtrees {
        for subtree in subtrees {
            let (subtree_num_robust, subtree_num_non_robust) = node_count(subtree.tree);
            num_robust += subtree_num_robust;
            num_non_robust += subtree_num_non_robust;
        }
    }

    (num_robust, num_non_robust)
}

pub fn robustness_accuracy<D: Dataset + Sync, S: Sample + Sync>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    test_data: Vec<S>,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split: usize,
) {

    let mut rng = thread_rng();

    let seed = rng.next_u64();

    for epsilon_factor in &[10000, 5000, 1000, 500, 100, 50] {
        let training_samples = samples.clone();
        let epsilon = 1.0 / *epsilon_factor as f64;

        let trees = ExtremelyRandomizedTrees::fit_with_epsilon(
            &dataset,
            training_samples,
            seed,
            num_trees,
            min_leaf_size,
            max_tries_per_split.clone(),
            epsilon
        );

        let mut t_p = 0;
        let mut t_n = 0;

        for sample in test_data.iter() {
            let predicted_label = trees.predict(sample);

            if sample.true_label() {
                if predicted_label {
                    t_p += 1;
                }
            } else {
                if !predicted_label {
                    t_n += 1;
                }
            }
        }

        let accuracy = (t_p + t_n) as f64 / test_data.len() as f64;
        println!("{},{},{}", name, epsilon_factor, accuracy);
    }


}

#[derive(Clone)]
enum Request<S: Sample> {
    Predict(S),
    Forget(S),
}

pub fn stress_test<D: Dataset + Sync, S: Sample + Sync>(
    name: &str,
    dataset: D,
    samples: Vec<S>,
    test_data: Vec<S>,
    multiplication_factor: usize,
    num_trees: usize,
    min_leaf_size: usize,
    max_tries_per_split: usize,
) {
    let mut rng = thread_rng();

    let target_robustness = ((dataset.num_records() as f64) / 1000.0).round() as usize;

    let mut stress_test_data = Vec::new();
    for _ in 0..multiplication_factor {
        for sample in &test_data {
            stress_test_data.push(Request::Predict(sample.clone()));
        }
    }

    stress_test_data.shuffle(&mut rng);

    let mut stress_test_data_with_forgets = stress_test_data.clone();

    (0..target_robustness)
        .for_each(|_| {
            let sample_to_forget_index = rng.gen_range(0, samples.len());
            let sample_to_forget = samples.get(sample_to_forget_index).unwrap().clone();
            let forget_request = Request::Forget(sample_to_forget);

            let forget_request_index = rng.gen_range(0, stress_test_data_with_forgets.len());
            let request = stress_test_data_with_forgets.get_mut(forget_request_index).unwrap();
            *request = forget_request;
        });

    let seed = rng.next_u64();

    let mut trees = ExtremelyRandomizedTrees::fit(
        &dataset,
        samples,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split.clone()
    );

    let prediction_start = Instant::now();
    for test_sample in &stress_test_data {
        match test_sample {
            Request::Predict(sample) => { trees.predict(sample); } ,
            Request::Forget(sample) => trees.forget(sample),
        }
    }
    let prediction_duration = prediction_start.elapsed();
    let throughput =
        ((stress_test_data.len() as f64 / prediction_duration.as_millis() as f64)
            * 1000.0) as usize;

    println!("{},predict_only,{},{}", name, prediction_duration.as_millis(), throughput);

    let prediction_start = Instant::now();
    for test_sample in &stress_test_data_with_forgets {
        match test_sample {
            Request::Predict(sample) => { trees.predict(sample); } ,
            Request::Forget(sample) => trees.forget(sample),
        }
    }
    let prediction_duration = prediction_start.elapsed();
    let throughput =
        ((stress_test_data_with_forgets.len() as f64 / prediction_duration.as_millis() as f64)
            * 1000.0) as usize;

    println!("{},forgets,{},{}", name, prediction_duration.as_millis(), throughput);
}
