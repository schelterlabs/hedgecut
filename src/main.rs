extern crate csv;
extern crate rand;

mod split_stats;
mod dataset;

use split_stats::SplitStats;
use dataset::{Dataset, TitanicDataset};

use rand::seq::SliceRandom;
use rand::Rng;

struct SplitCandidate {
    pub attribute_index: u8,
    pub cut_off: u8,
}

impl SplitCandidate {
    pub fn new(attribute_index: u8, cut_off: u8) -> SplitCandidate {
        SplitCandidate { attribute_index, cut_off }
    }
}




fn main() {

    let dataset = TitanicDataset::from_csv();

    let record_ids_to_split: Vec<u32> = (0 .. dataset.num_records()).collect();

    determine_split(record_ids_to_split, &dataset, 0, Vec::new());
}

fn determine_split<D: Dataset>(
    record_ids_to_split: Vec<u32>,
    dataset: &D,
    num_tries: u8,
    constant_attribute_indexes: Vec<u8>
) {

    let mut rng = rand::thread_rng();

    let mut attribute_indexes: Vec<u8> = (0..dataset.num_attributes())
        // TODO This searches linearly, but does it matter here?
        .filter(|attribute_index| !constant_attribute_indexes.contains(attribute_index))
        .collect();

    attribute_indexes.shuffle(&mut rng);

    // TODO can we allocate once and reuse the vec somehow?
    let split_candidates: Vec<SplitCandidate> = attribute_indexes.iter()
        .take(3)
        .map(|attribute_index| {
            let random_cut_off = rng.gen_range(0_u8, 20_u8);

            SplitCandidate::new(*attribute_index, /*random_attribute,*/ random_cut_off)
        })
        .collect();

    //TODO we could try to check the split attribute first (if we have it again)
    //TODO as it might be still in caches

    let split_stats: Vec<SplitStats> = split_candidates.iter()
        .map(|candidate| {
            compute_split_stats(&record_ids_to_split, &candidate, dataset, &dataset.labels())
        })
        .collect();

    let maybe_best_split_stats = split_stats.iter().enumerate()
        .filter(|(_, stats)| stats.score.is_some())
        .max_by(|(_, stats1), (_, stats2)| stats1.score.unwrap().cmp(&stats2.score.unwrap()));

    if maybe_best_split_stats.is_none() {
        if num_tries < 25 {
            determine_split(
                record_ids_to_split,
                dataset,
                num_tries + 1,
                constant_attribute_indexes.clone() // TODO try to get rid of the clone here
            );

            return;
        } else {
            //TODO Handle this case
            println!("Did not find a working split for {} records", record_ids_to_split.len());
            return;
        }
    }

    let (index_of_best_stats, best_split_stats) = maybe_best_split_stats.unwrap();

    let best_split_candidate = split_candidates.get(index_of_best_stats).unwrap();

    //println!("Best split {:?}", best_split_stats);

    let (record_ids_left, constant_value_on_the_left,
        record_ids_right, constant_value_on_the_right) = split(
        record_ids_to_split,
        dataset,
        &best_split_candidate,
        &best_split_stats);

    let min_leaf_size: usize = 3;
    // TODO move into conditional block
    let label_constant_on_the_left =
        best_split_stats.num_minus_left == 0 || best_split_stats.num_plus_left == 0;

    if record_ids_left.len() <= min_leaf_size || label_constant_on_the_left {
        println!("Building leaf for {} records", record_ids_left.len());
    } else {

        // TODO get rid of the clone here
        let mut constant_attribute_indexes_left = constant_attribute_indexes.clone();
        if constant_value_on_the_left {
            println!("Constant attribute found in {} records", record_ids_left.len());
            constant_attribute_indexes_left.push(best_split_candidate.attribute_index)
        }

        determine_split(record_ids_left, dataset, 0, constant_attribute_indexes_left);
    }

    // TODO move into conditional block
    let label_constant_on_the_right =
        best_split_stats.num_minus_right == 0 || best_split_stats.num_plus_right == 0;

    if record_ids_right.len() <= min_leaf_size || label_constant_on_the_right {
        println!("Building leaf for {} records", record_ids_right.len());
    } else {

        // TODO get rid of the clone here
        let mut constant_attribute_indexes_right = constant_attribute_indexes.clone();
        if constant_value_on_the_right {
            println!("Constant attribute found in {} records", record_ids_right.len());
            constant_attribute_indexes_right.push(best_split_candidate.attribute_index)
        }

        determine_split(record_ids_right, dataset, 0, constant_attribute_indexes_right);
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
        // TODO Remove boundary checks here later
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

