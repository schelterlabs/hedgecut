use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

use std::marker::Sync;

use crate::split_stats::{SplitStats, is_robust};
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
    pub fn fit<D>(
        dataset: &D,
        num_trees: usize
    ) -> ExtremelyRandomizedTrees
        where D: Dataset + Sync
    {

        let trees: Vec<Tree> = (0..num_trees)
            .into_par_iter()
            .map(|_| Tree::fit(dataset))
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
    elements: HashMap<u32, TreeElement>
}



impl Tree {

    fn fit<D: Dataset>(dataset: &D) -> Tree {

        // TODO replace with a faster hashmap
        let mut tree_elements: HashMap<u32, TreeElement> = HashMap::new();
        let record_ids_to_split: Vec<u32> = (0 .. dataset.num_records()).collect();

        //TODO try Cow for the constant attribute indexes
        determine_split(&mut tree_elements, record_ids_to_split, dataset, 1, 0, Vec::new());

        Tree { elements: tree_elements }
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

            let element = self.elements.get(&element_id).unwrap();

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

}

fn generate_random_split_candidates<D: Dataset>(
    dataset: &D,
    constant_attribute_indexes: &Vec<u8>
) -> Vec<SplitCandidate> {

    let mut rng = rand::thread_rng();

    let mut attribute_indexes: Vec<u8> = (0..dataset.num_attributes())
        // TODO This searches linearly, but does it matter here?
        .filter(|attribute_index| !constant_attribute_indexes.contains(attribute_index))
        .collect();

    attribute_indexes.shuffle(&mut rng);

    // TODO can we allocate once and reuse the vec somehow?
    let split_candidates: Vec<SplitCandidate> = attribute_indexes.iter()
        .take(5)// TODO should depend on number of attributes in dataset
        .map(|attribute_index| {

            let (min_value, max_value) = dataset.attribute_range(*attribute_index);

            let random_cut_off = rng.gen_range(min_value, max_value + 1);

            SplitCandidate::new(*attribute_index, random_cut_off)
        })
        .collect();

    split_candidates
}

fn determine_split<D: Dataset>(
    tree_elements: &mut HashMap<u32, TreeElement>,
    record_ids_to_split: Vec<u32>,
    dataset: &D,
    current_id: u32,
    num_tries: u8,
    constant_attribute_indexes: Vec<u8>
) {
    let max_num_tries = 25; // TODO make configurable

    let split_candidates = generate_random_split_candidates(dataset, &constant_attribute_indexes);

    //TODO we could try to check the split attribute first (if we have it again)
    //TODO as it might be still in caches

    let split_stats: Vec<SplitStats> = split_candidates.iter()
        .map(|candidate| {
            compute_split_stats(&record_ids_to_split, &candidate, dataset, &dataset.labels())
        })
        .collect();

    let maybe_best_split_stats = split_stats.iter().enumerate()
        .filter(|(_, stats)| stats.score != 0)
        .max_by(|(_, stats1), (_, stats2)| stats1.score.cmp(&stats2.score));

    if maybe_best_split_stats.is_none() {
        if num_tries < max_num_tries { // TODO make configurable
            determine_split(
                tree_elements,
                record_ids_to_split,
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

            tree_elements.insert(current_id, leaf);

            return;
        }
    }

    let (index_of_best_stats, best_split_stats) = maybe_best_split_stats.unwrap();

    let best_split_candidate = split_candidates.get(index_of_best_stats).unwrap();

    let mut at_least_one_non_robust = false;

    // TODO we might need to check all and proceed
    for (index, stats) in split_stats.iter().enumerate() {
        if index != index_of_best_stats {
            if !is_robust(best_split_stats, stats, 25) { // TODO make configurable
                at_least_one_non_robust = true;
                break;
            }
        }
    }

    if at_least_one_non_robust && num_tries < max_num_tries {
        //println!("Non-robust split found, retrying...");
        determine_split(
            tree_elements,
            record_ids_to_split,
            dataset,
            current_id,
            num_tries + 1,
            constant_attribute_indexes.clone() // TODO try to get rid of the clone here
        );
    } else {

        if at_least_one_non_robust {
            println!("Encountered non-robust split on {} records.", record_ids_to_split.len());
        }

        split_and_continue(
            tree_elements,
            record_ids_to_split,
            dataset,
            current_id,
            constant_attribute_indexes,
            best_split_candidate,
            best_split_stats
        );
    }
}

fn split_and_continue<D: Dataset>(
    tree_elements: &mut HashMap<u32, TreeElement>,
    record_ids_to_split: Vec<u32>,
    dataset: &D,
    current_id: u32,
    constant_attribute_indexes: Vec<u8>,
    best_split_candidate: &SplitCandidate,
    best_split_stats: &SplitStats
) {

    let (record_ids_left, constant_value_on_the_left,
        record_ids_right, constant_value_on_the_right) = split(
        record_ids_to_split,
        dataset,
        best_split_candidate,
        best_split_stats);

    let min_leaf_size: usize = 3; // TODO make configurable

    // TODO move into conditional block
    let label_constant_on_the_left =
        best_split_stats.num_minus_left == 0 || best_split_stats.num_plus_left == 0;

    let node = Tree::node(best_split_candidate.attribute_index, best_split_candidate.cut_off);

    tree_elements.insert(current_id, node);

    let left_child_id = current_id * 2;

    if record_ids_left.len() <= min_leaf_size || label_constant_on_the_left {
        //println!("Building leaf for {} records", record_ids_left.len());

        let leaf = Tree::leaf(
            best_split_stats.num_plus_left + best_split_stats.num_minus_left,
            best_split_stats.num_plus_left
        );

        tree_elements.insert(left_child_id, leaf);

    } else {

        // TODO get rid of the clone here
        let mut constant_attribute_indexes_left = constant_attribute_indexes.clone();
        if constant_value_on_the_left {
            //println!("Constant attribute found in {} records", record_ids_left.len());
            constant_attribute_indexes_left.push(best_split_candidate.attribute_index)
        }

        determine_split(
            tree_elements,
            record_ids_left,
            dataset,
            left_child_id,
            0,
            constant_attribute_indexes_left
        );
    }

    // TODO move into conditional block
    let label_constant_on_the_right =
        best_split_stats.num_minus_right == 0 || best_split_stats.num_plus_right == 0;

    let right_child_id = (current_id * 2) + 1;

    if record_ids_right.len() <= min_leaf_size || label_constant_on_the_right {
        //println!("Building leaf for {} records", record_ids_right.len());

        let leaf = Tree::leaf(
            best_split_stats.num_plus_right + best_split_stats.num_minus_right,
            best_split_stats.num_plus_right
        );

        tree_elements.insert(right_child_id, leaf);

    } else {

        // TODO get rid of the clone here
        let mut constant_attribute_indexes_right = constant_attribute_indexes.clone();
        if constant_value_on_the_right {
            //println!("Constant attribute found in {} records", record_ids_right.len());
            constant_attribute_indexes_right.push(best_split_candidate.attribute_index)
        }

        determine_split(
            tree_elements,
            record_ids_right,
            dataset,
            right_child_id,
            0,
            constant_attribute_indexes_right
        );
    }
}

fn compute_split_stats<D: Dataset>(
    record_ids_to_split: &Vec<u32>,
    split_candidate: &SplitCandidate,
    dataset: &D,
    labels: &Vec<bool>
) -> SplitStats {
    let mut split_stats = SplitStats::new();

    let attribute_to_split_on = dataset.attribute(split_candidate.attribute_index);
    let cut_off = split_candidate.cut_off;

    for record_id in record_ids_to_split {
        // TODO Maybe remove boundary checks here later
        let plus = *labels.get(*record_id as usize).unwrap();

        let attribute_value = *attribute_to_split_on.get(*record_id as usize).unwrap();
        let is_left =  attribute_value < cut_off;

        split_stats.update(plus, is_left);
    }

    split_stats.update_score();

    split_stats
}

fn split<D: Dataset>(
    record_ids_to_split: Vec<u32>,
    dataset: &D,
    split_candidate: &SplitCandidate,
    split_stats: &SplitStats
) -> (Vec<u32>, bool, Vec<u32>, bool) { // TODO replace with a struct for readability

    let attribute_to_split_on = dataset.attribute(split_candidate.attribute_index);
    let cut_off = split_candidate.cut_off;

    // TODO We copy here for safety, should reuse the allocated vector later
    let num_left = split_stats.num_plus_left + split_stats.num_minus_left;
    let mut record_ids_left = Vec::with_capacity(num_left as usize);

    let num_right = split_stats.num_plus_right + split_stats.num_minus_right;
    let mut record_ids_right = Vec::with_capacity(num_right as usize);

    let mut constant_value_on_the_left = true;
    let mut first_value_on_the_left: Option<u8> = None;
    let mut constant_value_on_the_right = true;
    let mut first_value_on_the_right: Option<u8> = None;

    for record_id in record_ids_to_split {

        let attribute_value = *attribute_to_split_on.get(record_id as usize).unwrap();
        let is_left =  attribute_value < cut_off;

        if is_left {
            if first_value_on_the_left.is_none() {
                first_value_on_the_left = Some(attribute_value);
            } else {
                if constant_value_on_the_left &&
                    attribute_value != first_value_on_the_left.unwrap() {
                    constant_value_on_the_left = false;
                }
            }

            record_ids_left.push(record_id);
        } else {

            if first_value_on_the_right.is_none() {
                first_value_on_the_right = Some(attribute_value);
            } else {
                if constant_value_on_the_right &&
                    attribute_value != first_value_on_the_right.unwrap() {
                    constant_value_on_the_right = false;
                }
            }

            record_ids_right.push(record_id);
        }
    }

    (record_ids_left, constant_value_on_the_left, record_ids_right, constant_value_on_the_right)
}