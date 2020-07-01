#![allow(deprecated)] // TODO use rand_xorshift crate to remove this...
use rand::{Rng,SeedableRng,XorShiftRng};
use rand::seq::SliceRandom;
use rayon::prelude::*;

use std::marker::Sync;
use hashbrown::HashMap;


use crate::split_stats::SplitStats;//, is_robust};
use crate::dataset::{Dataset, Sample};

struct SplitCandidate {
    pub attribute_index: u8,
    pub cut_off: u8,
}

impl SplitCandidate {
    pub fn new(attribute_index: u8, cut_off: u8) -> SplitCandidate {
        SplitCandidate { attribute_index, cut_off }
    }
}

pub struct ExtremelyRandomizedTrees {
    trees: Vec<Tree>,
}

impl ExtremelyRandomizedTrees {

    pub fn fit<D, S>(
        dataset: &D,
        samples: Vec<S>,
        seed: u64,
        num_trees: usize,
        min_leaf_size: usize,
        max_tries_per_split: usize,
    ) -> ExtremelyRandomizedTrees
        where D: Dataset + Sync, S: Sample + Sync
    {

        let num_attributes_to_try_per_split =
            (dataset.num_attributes() as f64).sqrt().round() as usize;

        let target_robustness = ((dataset.num_records() as f64) / 1000.0).round() as usize;

        println!(
            "Fitting {} trees on {} records with num_attributes_to_try_per_split={}, \
             target_robustness={}, max_tries_per_split={}",
            num_trees,
            dataset.num_records(),
            num_attributes_to_try_per_split,
            target_robustness,
            max_tries_per_split
        );

        let trees: Vec<Tree> = (0..num_trees)
            .into_par_iter()
            .map(|tree_index| Tree::fit(
                dataset,
                samples.clone().as_mut_slice(),
                seed,
                tree_index as u64,
                min_leaf_size,
                num_attributes_to_try_per_split,
                target_robustness,
                max_tries_per_split
            ))
            .collect();

        ExtremelyRandomizedTrees { trees }
    }

    pub fn predict<S>(
        &self,
        sample: &S
    ) -> bool
        where S: Sample + Sync
    {
        let num_plus: usize = self.trees
            .par_iter()
            .filter_map(|tree| {
                let is_plus = tree.predict(sample);
                if is_plus { Some(1_usize) } else { None }
            })
            .sum();

        num_plus * 2 > self.trees.len()
    }
}


#[derive(Eq,PartialEq,Debug)]
enum TreeElement {
    Node { attribute_index: u8, cut_off: u8 },
    Leaf { num_samples: u32, num_plus: u32 }
}

struct Tree {
    rng: XorShiftRng,
    tree_elements: HashMap<u32, TreeElement>,
    min_leaf_size: usize,
    num_attributes_to_try_per_split: usize,
    target_robustness: usize,
    max_tries_per_split: usize,
}



impl Tree {

    fn fit<D: Dataset, S: Sample>(
        dataset: &D,
        samples: &mut [S],
        seed: u64,
        tree_index: u64,
        min_leaf_size: usize,
        num_attributes_to_try_per_split: usize,
        target_robustness: usize,
        max_tries_per_split: usize
    ) -> Tree {

        let rng = XorShiftRng::from_seed(as_bytes(seed, tree_index));

        let mut tree = Tree {
            rng,
            tree_elements: HashMap::new(),
            min_leaf_size,
            num_attributes_to_try_per_split,
            target_robustness,
            max_tries_per_split,
        };

        let p_plus = dataset.num_plus() as f64 / dataset.num_records() as f64;
        let p_minus = (dataset.num_records() - dataset.num_plus()) as f64 /
            dataset.num_records() as f64;

        let gini_initial = 1.0 - (p_plus * p_plus) - (p_minus * p_minus);

        //TODO try Cow for the constant attribute indexes
        tree.determine_split(gini_initial, samples, dataset, 1, 0, Vec::new());

        return tree;
    }

    fn leaf(num_samples: u32, num_plus: u32) -> TreeElement {
        TreeElement::Leaf { num_samples, num_plus }
    }

    fn node(attribute_index: u8, cut_off: u8) -> TreeElement {
        TreeElement::Node { attribute_index, cut_off }
    }

    fn predict<S: Sample>(&self, sample: &S) -> bool {

        let mut element_id = 1;

        loop {

            let element = self.tree_elements.get(&element_id).unwrap();

            match element {

                TreeElement::Node { attribute_index, cut_off } => {
                    if sample.is_smaller_than(*attribute_index, *cut_off) {
                        element_id = element_id * 2;
                    } else {
                        element_id = (element_id * 2) + 1;
                    }
                }

                TreeElement::Leaf { num_samples, num_plus } => {
                    return (*num_plus * 2) > *num_samples;
                }
            }
        }
    }

    fn generate_random_split_candidates<D: Dataset>(
        &mut self,
        dataset: &D,
        constant_attribute_indexes: &Vec<u8>
    ) -> Vec<SplitCandidate> {

        let mut attribute_indexes: Vec<u8> = (0..dataset.num_attributes())
            // TODO This searches linearly, but does it matter here?
            .filter(|attribute_index| !constant_attribute_indexes.contains(attribute_index))
            .collect();

        attribute_indexes.shuffle(&mut self.rng);



        // TODO can we allocate once and reuse the vec somehow?
        let split_candidates: Vec<SplitCandidate> = attribute_indexes.iter()
            .take(self.num_attributes_to_try_per_split)
            .map(|attribute_index| {

                let (min_value, max_value) = dataset.attribute_range(*attribute_index);

                let random_cut_off = self.rng.gen_range(min_value, max_value + 1);

                SplitCandidate::new(*attribute_index, random_cut_off)
            })
            .collect();

        split_candidates
    }

    fn determine_split<D: Dataset, S: Sample>(
        &mut self,
        impurity_before: f64,
        samples: &mut [S],
        dataset: &D,
        current_id: u32,
        num_tries: usize,
        constant_attribute_indexes: Vec<u8>
    ) {
        let split_candidates =
            self.generate_random_split_candidates(dataset, &constant_attribute_indexes);


        let split_stats: Vec<SplitStats> = split_candidates.iter()
            .map(|candidate| compute_split_stats(impurity_before, &samples, &candidate))
            .collect();

        let maybe_best_split_stats = split_stats.iter().enumerate()
            .filter(|(_, stats)| stats.score != 0)
            .max_by(|(_, stats1), (_, stats2)| stats1.score.cmp(&stats2.score));

        if maybe_best_split_stats.is_none() {
            if num_tries < self.max_tries_per_split {
                self.determine_split(
                    impurity_before,
                    samples,
                    dataset,
                    current_id,
                    num_tries + 1,
                    constant_attribute_indexes.clone() // TODO try to get rid of the clone here
                );

                return;
            } else {

                let some_stats = split_stats.first().unwrap();
                let num_plus = some_stats.num_plus_left + some_stats.num_plus_right;
                let num_samples = some_stats.num_minus_left + some_stats.num_minus_right + num_plus;

                let leaf = Tree::leaf(num_samples, num_plus);

                self.tree_elements.insert(current_id, leaf);

                return;
            }
        }

        let (index_of_best_stats, best_split_stats) = maybe_best_split_stats.unwrap();

        let best_split_candidate = split_candidates.get(index_of_best_stats).unwrap();

        let mut at_least_one_non_robust = false;

        // // TODO we might need to check all and proceed
        // for (index, stats) in split_stats.iter().enumerate() {
        //     if index != index_of_best_stats {
        //         if !is_robust(best_split_stats, stats, self.target_robustness) {
        //             at_least_one_non_robust = true;
        //             break;
        //         }
        //     }
        // }

        if at_least_one_non_robust && num_tries < self.max_tries_per_split {
            //println!("Non-robust split found, retrying...");
            self.determine_split(
                impurity_before,
                samples,
                dataset,
                current_id,
                num_tries + 1,
                constant_attribute_indexes.clone() // TODO try to get rid of the clone here
            );
        } else {

            if at_least_one_non_robust {
                println!("Encountered non-robust split on {} records.", samples.len());
            }

            self.split_and_continue(
                samples,
                dataset,
                current_id,
                constant_attribute_indexes,
                best_split_candidate,
                best_split_stats
            );
        }
    }

    fn split_and_continue<D: Dataset, S: Sample>(
        &mut self,
        samples: &mut [S],
        dataset: &D,
        current_id: u32,
        constant_attribute_indexes: Vec<u8>,
        best_split_candidate: &SplitCandidate,
        best_split_stats: &SplitStats
    ) {

        let (samples_left, constant_on_the_left, samples_right, constant_on_the_right) =
            split(samples, best_split_candidate);

        let node = Tree::node(best_split_candidate.attribute_index, best_split_candidate.cut_off);

        self.tree_elements.insert(current_id, node);

        let left_child_id = current_id * 2;

        let label_constant_on_the_left =
            best_split_stats.num_minus_left == 0 || best_split_stats.num_plus_left == 0;

        if samples_left.len() <= self.min_leaf_size || label_constant_on_the_left {
            //println!("Building leaf for {} records", record_ids_left.len());

            let leaf = Tree::leaf(
                best_split_stats.num_plus_left + best_split_stats.num_minus_left,
                best_split_stats.num_plus_left
            );

            self.tree_elements.insert(left_child_id, leaf);

        } else {

            // TODO get rid of the clone here
            let mut constant_attribute_indexes_left = constant_attribute_indexes.clone();
            if constant_on_the_left {
                //println!("Constant attribute found in {} records", record_ids_left.len());
                constant_attribute_indexes_left.push(best_split_candidate.attribute_index)
            }

            self.determine_split(
                best_split_stats.impurity_left,
                samples_left,
                dataset,
                left_child_id,
                0,
                constant_attribute_indexes_left
            );
        }

        let right_child_id = (current_id * 2) + 1;

        let label_constant_on_the_right =
            best_split_stats.num_minus_right == 0 || best_split_stats.num_plus_right == 0;

        if samples_right.len() <= self.min_leaf_size || label_constant_on_the_right {
            //println!("Building leaf for {} records", record_ids_right.len());

            let leaf = Tree::leaf(
                best_split_stats.num_plus_right + best_split_stats.num_minus_right,
                best_split_stats.num_plus_right
            );

            self.tree_elements.insert(right_child_id, leaf);

        } else {

            // TODO get rid of the clone here
            let mut constant_attribute_indexes_right = constant_attribute_indexes.clone();
            if constant_on_the_right {
                //println!("Constant attribute found in {} records", record_ids_right.len());
                constant_attribute_indexes_right.push(best_split_candidate.attribute_index)
            }

            self.determine_split(
                best_split_stats.impurity_right,
                samples_right,
                dataset,
                right_child_id,
                0,
                constant_attribute_indexes_right
            );
        }
    }
}


fn compute_split_stats<S: Sample>(
    impurity_before: f64,
    samples: &[S],
    split_candidate: &SplitCandidate,
) -> SplitStats {
    let mut split_stats = SplitStats::new();

    for sample in samples {

        let plus = sample.true_label();

        let is_left =
            sample.is_smaller_than(split_candidate.attribute_index, split_candidate.cut_off);

        split_stats.update(plus, is_left);
    }

    split_stats.update_score(impurity_before);

    split_stats
}

// TODO needs to be tested more thoroughly
fn split<'a, S>(
    samples: &'a mut [S],
    split_candidate: &SplitCandidate
) -> (&'a mut [S], bool, &'a mut [S], bool)
    where S: Sample
{

    let cut_off = split_candidate.cut_off;

    let mut cursor = 0;
    let mut cursor_end = samples.len();

    let mut constant_on_the_left = true;
    let mut first_value_on_the_left: Option<u8> = None;
    let mut constant_on_the_right = true;
    let mut first_value_on_the_right: Option<u8> = None;

    loop {
        // TODO Maybe remove boundary checks here later

        let sample = samples.get(cursor).unwrap();
        let attribute_value: u8 = sample.attribute_value(split_candidate.attribute_index);

        if attribute_value < cut_off {

            if constant_on_the_left {
                if first_value_on_the_left.is_none() {
                    first_value_on_the_left = Some(attribute_value);
                } else if attribute_value != first_value_on_the_left.unwrap() {
                    constant_on_the_left = false;
                }
            }

            cursor += 1;
        } else {

            if constant_on_the_right {
                if first_value_on_the_right.is_none() {
                    first_value_on_the_right = Some(attribute_value);
                } else if attribute_value != first_value_on_the_right.unwrap() {
                    constant_on_the_right = false;
                }
            }

            cursor_end -= 1;
            //println!("Swapping {} and {} with record {}({}), cursor_end is now {}",
            // cursor, cursor_end, record_id, value, cursor_end);
            samples.swap(cursor, cursor_end);
        }

        if cursor == cursor_end - 1 {
            break;
        }
    }

    let (samples_left, samples_right) = samples.split_at_mut(cursor);

    (samples_left, constant_on_the_left, samples_right, constant_on_the_right)
}

fn as_bytes(seed: u64, tree_index: u64) -> [u8; 16] {

    use std::mem::transmute;
    let seed_bytes: [u8; 8] = unsafe { transmute(seed.to_be()) };
    let tree_index_bytes: [u8; 8] = unsafe { transmute(tree_index.to_be()) };

    [
        *seed_bytes.get(0).unwrap(),
        *seed_bytes.get(1).unwrap(),
        *seed_bytes.get(2).unwrap(),
        *seed_bytes.get(3).unwrap(),
        *seed_bytes.get(4).unwrap(),
        *seed_bytes.get(5).unwrap(),
        *seed_bytes.get(6).unwrap(),
        *seed_bytes.get(7).unwrap(),
        *tree_index_bytes.get(0).unwrap(),
        *tree_index_bytes.get(1).unwrap(),
        *tree_index_bytes.get(2).unwrap(),
        *tree_index_bytes.get(3).unwrap(),
        *tree_index_bytes.get(4).unwrap(),
        *tree_index_bytes.get(5).unwrap(),
        *tree_index_bytes.get(6).unwrap(),
        *tree_index_bytes.get(7).unwrap(),
    ]
}