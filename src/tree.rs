#![allow(deprecated)] // TODO use rand_xorshift crate to remove this...
use rand::{Rng, SeedableRng, XorShiftRng};
use rand::seq::SliceRandom;
use rayon::prelude::*;

use std::marker::Sync;
use std::borrow::Cow;
use hashbrown::HashMap;

use crate::scan::{scan, scan_simd};
use crate::utils::as_bytes;

use crate::split_stats::{SplitStats, is_robust};
use crate::dataset::{Dataset, Sample, AttributeType};

#[derive(Eq,PartialEq,Clone,Debug)]
pub enum Split {
    Numerical { attribute_index: u8, cut_off: u8 },
    Categorical { attribute_index: u8, subset: u64 }
}


impl Split {
    pub fn new_numerical(attribute_index: u8, cut_off: u8) -> Split {
        Split::Numerical { attribute_index, cut_off }
    }

    pub fn new_categorical(attribute_index: u8, subset: u64) -> Split {
        Split::Categorical { attribute_index, subset }
    }

    pub fn attribute_index(&self) -> u8 {
        match self {
            Split::Numerical { attribute_index, cut_off: _ } => *attribute_index,
            Split::Categorical { attribute_index, subset: _ } => *attribute_index,
        }
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
            .filter(|tree| tree.predict(sample))
            .count();

        num_plus * 2 > self.trees.len()
    }

    pub fn forget<S>(&mut self, sample: &S) where S: Sample + Sync {
        self.trees.par_iter_mut().for_each(|tree| { tree.forget(sample); })
    }
}


#[derive(Eq,PartialEq,Debug)]
enum TreeElement {
    Node { split: Split },
    Leaf { num_samples: u32, num_plus: u32 }
}

struct Tree {
    index: usize,
    rng: XorShiftRng,
    tree_elements: HashMap<u64, TreeElement>,
    alternatives: HashMap<u64, Vec<AlternativeTree>>,
    min_leaf_size: usize,
    num_attributes_to_try_per_split: usize,
    max_tries_per_split: usize,
}

struct AlternativeTree {
    split_candidate: Split,
    split_stats: SplitStats,
    tree: Tree,
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
            index: tree_index as usize,
            rng,
            tree_elements: HashMap::new(),
            alternatives: HashMap::new(),
            min_leaf_size,
            num_attributes_to_try_per_split,
            max_tries_per_split,
        };

        // TODO this should use code from split_stats
        let p_plus = dataset.num_plus() as f64 / dataset.num_records() as f64;

        let gini_initial = 1.0 - (p_plus * p_plus);

        let mut constant_attribute_indexes: Cow<[u8]> = Cow::from(Vec::new());

        tree.determine_split(
            gini_initial,
            target_robustness,
            samples,
            dataset,
            1,
            0,
            &mut constant_attribute_indexes);

        return tree;
    }

    fn leaf(num_samples: u32, num_plus: u32) -> TreeElement {
        TreeElement::Leaf { num_samples, num_plus }
    }

    fn node(split: Split) -> TreeElement {
        TreeElement::Node { split }
    }

    fn predict<S: Sample>(&self, sample: &S) -> bool {

        let mut current_tree = self;
        let mut element_id = 1;

        loop {

            let element = current_tree.tree_elements.get(&element_id);

            match element {

                Some(TreeElement::Node { split }) => {
                    if sample.is_left_of(split) {
                        element_id = element_id * 2;
                    } else {
                        element_id = (element_id * 2) + 1;
                    }
                }

                Some(TreeElement::Leaf { num_samples, num_plus }) => {
                    return (*num_plus * 2) > *num_samples;
                }

                None => {
                    let alternative_trees = current_tree.alternatives.get(&element_id).unwrap();
                    // First tree in this list is the current best one by definition
                    current_tree = &alternative_trees.first().unwrap().tree;
                }
            }
        }
    }

    fn forget<S: Sample>(&mut self, sample: &S) {
        Tree::forget_from(self, sample, 1);
    }

    fn forget_from<S: Sample>(tree: &mut Tree, sample: &S, element_id_to_start: u64) {

        let mut element_id = element_id_to_start;

        loop {

            let element = tree.tree_elements.get(&element_id);

            match element {

                Some(TreeElement::Node { split }) => {

                    if sample.is_left_of(split) {
                        element_id = element_id * 2;
                    } else {
                        element_id = (element_id * 2) + 1;
                    }
                }

                Some(TreeElement::Leaf { num_samples, num_plus }) => {

                    let new_num_samples = num_samples - 1;
                    let new_num_plus = if sample.true_label() {
                        num_plus - 1
                    } else {
                        *num_plus
                    };

                    let updated_leaf = Tree::leaf(new_num_samples, new_num_plus);
                    tree.tree_elements.insert(element_id, updated_leaf);
                    break;
                }

                None => {
                    // We hit a non-robust node
                    println!("We hit a non-robust node, have to update statistics and check alternatives!");

                    // First we have to update the split stats
                    let alternative_trees = &mut *tree.alternatives.get_mut(&element_id).unwrap();

                    alternative_trees.iter_mut().for_each(|alternative_tree| {
                        let stats = &mut alternative_tree.split_stats;

                        if sample.is_left_of(&alternative_tree.split_candidate) {
                            if sample.true_label() {
                                stats.num_plus_left -= 1;
                            } else {
                                stats.num_minus_left -= 1;
                            }
                        } else {
                            if sample.true_label() {
                                stats.num_plus_right -= 1;
                            } else {
                                stats.num_minus_right -= 1;
                            }
                        }

                        stats.update_score_and_impurity_before();
                    });

                    // Then we resort to put the best tree in the first position
                    // TODO alternative_trees could be a heap, but it probably does not matter
                    // Make sure the split with the highest score is in the first position
                    alternative_trees.sort_by(|tree_a, tree_b| {
                        tree_b.split_stats.score.cmp(&tree_a.split_stats.score)
                    });

                    // Afterwards, we invoke the forgetting procedure on the alternative trees
                    alternative_trees.iter_mut().for_each(|alternative_tree| {
                        Tree::forget_from(&mut alternative_tree.tree, sample, element_id);
                    });

                    break;
                }
            }
        }
    }

    fn generate_split_candidates<D: Dataset>(
        &mut self,
        dataset: &D,
        constant_attribute_indexes: &Cow<[u8]>
    ) -> Vec<Split> {

        let mut attribute_indexes: Vec<u8> = (0..dataset.num_attributes())
            // TODO This searches linearly, but does it matter here?
            .filter(|attribute_index| !constant_attribute_indexes.contains(attribute_index))
            .collect();

        attribute_indexes.shuffle(&mut self.rng);



        // TODO can we allocate once and reuse the vec somehow?
        let split_candidates: Vec<Split> = attribute_indexes.iter()
            .take(self.num_attributes_to_try_per_split)
            .map(|attribute_index| generate_random_split(&mut self.rng, dataset, *attribute_index))
            .collect();

        split_candidates
    }




    fn determine_split<D: Dataset, S: Sample>(
        &mut self,
        impurity_before: f64,
        target_robustness: usize,
        samples: &mut [S],
        dataset: &D,
        current_id: u64,
        num_tries: usize,
        constant_attribute_indexes: &mut Cow<[u8]>
    ) {
        let split_candidates = self.generate_split_candidates(dataset, &constant_attribute_indexes);

        let split_stats = compute_split_stats(impurity_before, &samples, &split_candidates);

        let maybe_best_split_stats = split_stats.iter().enumerate()
            .filter(|(_, stats)| stats.score != 0)
            .max_by(|(_, stats1), (_, stats2)| stats1.score.cmp(&stats2.score));

        if maybe_best_split_stats.is_none() {
            if num_tries < self.max_tries_per_split {
                self.determine_split(
                    impurity_before,
                    target_robustness,
                    samples,
                    dataset,
                    current_id,
                    num_tries + 1,
                    constant_attribute_indexes
                );

                return;
            } else {

                // We only need stats that are indepent of the split
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
        let mut num_removals_required = 0;

        for (index, stats) in split_stats.iter().enumerate() {
            if index != index_of_best_stats {

                let (is_robust_split, num_removals_evaluated) =
                    is_robust(best_split_stats, stats, target_robustness);

                if !is_robust_split {
                    at_least_one_non_robust = true;
                    num_removals_required = num_removals_evaluated;
                    break;
                }
            }
        }

        if at_least_one_non_robust && num_tries < self.max_tries_per_split {
            //println!("Non-robust split found, retrying...");
            self.determine_split(
                impurity_before,
                target_robustness,
                samples,
                dataset,
                current_id,
                num_tries + 1,
                constant_attribute_indexes
            );
        } else {

            if at_least_one_non_robust {

                let mut alternative_splits: Vec<(usize, usize)> = split_stats.iter()
                    .enumerate()
                    .filter(|(index, _)| *index != index_of_best_stats)
                    .filter_map(|(index, stats)| {
                        let (is_robust_split, num_removals_required_to_break_split) =
                            is_robust(best_split_stats, stats, target_robustness);

                        if is_robust_split {
                            None
                        } else {
                            Some((index, num_removals_required_to_break_split))
                        }
                    })
                    .collect();

                println!(
                    "Non-robust split ({}) on {} records with {} alternatives for element_id {} in tree {}.",
                    num_removals_required,
                    samples.len(),
                    alternative_splits.len(),
                    current_id,
                    self.index
                );

                alternative_splits.push((index_of_best_stats, 0));

                let mut alternative_trees: Vec<AlternativeTree> =
                    Vec::with_capacity(alternative_splits.len());

                for (index, num_removals_required_to_break_split) in alternative_splits {

                    let alternative_target_robustness =
                        target_robustness - num_removals_required_to_break_split;

                    let mut copy_of_samples = samples.to_vec();

                    let replacement_tree = Tree {
                        index: self.index,
                        rng: self.rng.clone(),
                        tree_elements: HashMap::new(),
                        alternatives: HashMap::new(),
                        min_leaf_size: self.min_leaf_size,
                        num_attributes_to_try_per_split: self.num_attributes_to_try_per_split,
                        max_tries_per_split: self.max_tries_per_split
                    };

                    let alternative_split_candidate = split_candidates.get(index).unwrap();
                    let alternative_split_stats = split_stats.get(index).unwrap();

                    let mut alternative_tree = AlternativeTree {
                        split_candidate: alternative_split_candidate.clone(),
                        split_stats: alternative_split_stats.clone(),
                        tree: replacement_tree
                    };

                    alternative_tree.tree.split_and_continue(
                        alternative_target_robustness,
                        copy_of_samples.as_mut_slice(),
                        dataset,
                        current_id,
                        &mut constant_attribute_indexes.clone(),
                        alternative_split_candidate,
                        alternative_split_stats
                    );

                    alternative_trees.push(alternative_tree);
                }

                // TODO alternative_trees could be a heap, but it probably does not matter
                // Make sure the split with the highest score is in the first position
                alternative_trees.sort_by(|tree_a, tree_b| {
                    tree_b.split_stats.score.cmp(&tree_a.split_stats.score)
                });

                self.alternatives.insert(current_id, alternative_trees);

            } else {

                self.split_and_continue(
                    target_robustness,
                    samples,
                    dataset,
                    current_id,
                    constant_attribute_indexes,
                    best_split_candidate,
                    best_split_stats
                );
            }
        }
    }

    fn split_and_continue<D: Dataset, S: Sample>(
        &mut self,
        target_robustness: usize,
        samples: &mut [S],
        dataset: &D,
        current_id: u64,
        constant_attribute_indexes: &mut Cow<[u8]>,
        best_split: &Split,
        best_split_stats: &SplitStats
    ) {
        // println!(
        //     "current node {} on {},{}",
        //     current_id,
        //     best_split_stats.num_minus_left + best_split_stats.num_plus_left,
        //     best_split_stats.num_plus_right + best_split_stats.num_minus_right
        // );

        let (samples_left, constant_on_the_left, samples_right, constant_on_the_right) =
            split(samples, best_split);

        let node = Tree::node(best_split.clone());

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

            let mut constant_attribute_indexes_left = constant_attribute_indexes.clone();
            if constant_on_the_left {
                //println!("Constant attribute found in {} records", record_ids_left.len());
                let attribute_index = best_split.attribute_index();
                constant_attribute_indexes_left.to_mut().push(attribute_index);
            }

            self.determine_split(
                best_split_stats.impurity_left,
                target_robustness,
                samples_left,
                dataset,
                left_child_id,
                0,
                &mut constant_attribute_indexes_left
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

            let mut constant_attribute_indexes_right = constant_attribute_indexes.clone();
            if constant_on_the_right {
                //println!("Constant attribute found in {} records", record_ids_right.len());
                let attribute_index = best_split.attribute_index();
                constant_attribute_indexes_right.to_mut().push(attribute_index);
            }

            self.determine_split(
                best_split_stats.impurity_right,
                target_robustness,
                samples_right,
                dataset,
                right_child_id,
                0,
                &mut constant_attribute_indexes_right
            );
        }
    }
}

fn compute_split_stats<S: Sample>(
    impurity_before: f64,
    samples: &[S],
    split_candidates: &Vec<Split>,
) -> Vec<SplitStats> {

    let mut all_stats: Vec<SplitStats> = Vec::with_capacity(split_candidates.len());
    
    for candidate in split_candidates {

        let mut stats = match candidate {
            Split::Numerical { attribute_index: _, cut_off: _ } => {
                scan_simd(samples, candidate)
            },
            Split::Categorical { attribute_index: _, subset: _ } => {
                // TODO we also need a SIMD version here
                scan(samples, &candidate)
            },
        };

        stats.update_score(impurity_before);
        all_stats.push(stats);
    }

    all_stats
}

// TODO needs to be tested more thoroughly
fn split<'a, S: Sample>(
    samples: &'a mut [S],
    split: &Split
) -> (&'a mut [S], bool, &'a mut [S], bool) {

    //let cut_off = split_candidate.cut_off;

    let mut cursor = 0;
    let mut cursor_end = samples.len();

    let mut constant_on_the_left = true;
    let mut first_value_on_the_left: Option<u8> = None;
    let mut constant_on_the_right = true;
    let mut first_value_on_the_right: Option<u8> = None;

    loop {
        // TODO Maybe remove boundary checks here later

        let sample = samples.get(cursor).unwrap();
        let attribute_value: u8 = sample.attribute_value(split.attribute_index());

        if sample.is_left_of(&split) {

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

fn generate_random_split<D: Dataset>(
    rng: &mut XorShiftRng,
    dataset: &D,
    attribute_index: u8
) -> Split {
    match dataset.attribute_type(attribute_index) {
        AttributeType::Numerical => {
            let (min_value, max_value) = dataset.attribute_range(attribute_index);

            let random_cut_off = rng.gen_range(min_value, max_value + 1);

            Split::new_numerical(attribute_index, random_cut_off)
        },
        AttributeType::Categorical => {
            let (_, cardinality) = dataset.attribute_range(attribute_index);

            let how_many = rng.gen_range(0, cardinality + 1);
            // TODO lets get rid of the allocation here...
            let mut values: Vec<u8> = (0..(cardinality + 1)).collect();
            values.shuffle(rng);

            let mut subset: u64 = 0;
            for bit_to_set in values.iter().take(how_many as usize) {
                subset |= (1 << *bit_to_set) as u64
            }

            Split::new_categorical(attribute_index, subset)
        }
    }
}