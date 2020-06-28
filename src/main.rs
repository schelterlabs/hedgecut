extern crate csv;

use std::str::FromStr;

mod split_stats;

use split_stats::SplitStats;

struct SplitCandidate<'a> {
    pub attribute: &'a Vec<u8>,
    pub cut_off: u8,
}

impl SplitCandidate<'_> {
    pub fn new(attribute: &Vec<u8>, cut_off: u8) -> SplitCandidate {
        SplitCandidate { attribute, cut_off }
    }
}

struct TitanicDataset {
    pub num_records: u32,
    pub age: Vec<u8>,
    pub fare: Vec<u8>,
    pub siblings: Vec<u8>,
    pub children: Vec<u8>,
    pub gender: Vec<u8>,
    pub pclass: Vec<u8>,
    pub labels: Vec<bool>,
}

impl TitanicDataset {

    pub fn from_csv() -> TitanicDataset {
        let num_records = 886;

        let mut age: Vec<u8> = Vec::with_capacity(num_records);
        let mut fare: Vec<u8> = Vec::with_capacity(num_records);
        let mut siblings: Vec<u8> = Vec::with_capacity(num_records);
        let mut children: Vec<u8> = Vec::with_capacity(num_records);
        let mut gender: Vec<u8> = Vec::with_capacity(num_records);
        let mut pclass: Vec<u8> = Vec::with_capacity(num_records);

        let mut labels: Vec<bool> = Vec::with_capacity(num_records);

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b'\t')
            .from_path("titanic-attributes.csv")
            .unwrap();

        for result in reader.records() {
            let record = result.unwrap();

            let record_id: u32 = u32::from_str(record.get(0).unwrap()).unwrap();
            let attribute_name = record.get(1).unwrap();
            let attribute_value = u8::from_str(record.get(2).unwrap()).unwrap();

            match attribute_name {
                "age" => age.insert(record_id as usize, attribute_value),
                "fare" => fare.insert(record_id as usize, attribute_value),
                "siblings" => siblings.insert(record_id as usize, attribute_value),
                "children" => children.insert(record_id as usize, attribute_value),
                "gender" => gender.insert(record_id as usize, attribute_value),
                "pclass" => pclass.insert(record_id as usize, attribute_value),
                "label" => labels.insert(record_id as usize, attribute_value == 1),

                _ => println!("UNKNOWN ATTRIBUTE ENCOUNTERED")
            }
        }

        TitanicDataset {
            num_records: num_records as u32,
            age,
            fare,
            siblings,
            children,
            gender,
            pclass,
            labels
        }
    }

}


fn main() {

    let dataset = TitanicDataset::from_csv();

    let mut record_ids_to_split: Vec<u32> = Vec::with_capacity(dataset.num_records as usize);
    for record_id in 0 .. dataset.num_records {
        record_ids_to_split.push(record_id);
    }

    let split_candidates = vec![
        SplitCandidate::new(&dataset.age, 5),
        SplitCandidate::new(&dataset.fare, 12),
        SplitCandidate::new(&dataset.children, 3),
        SplitCandidate::new(&dataset.gender, 1),
    ];

    let split_stats: Vec<SplitStats> = split_candidates.iter()
        .map(|candidate| compute_split_stats(&record_ids_to_split, &candidate, &dataset.labels))
        .collect();

    let maybe_best_split_stats = split_stats.iter().enumerate()
        .filter(|(_, stats)| stats.score.is_some())
        .max_by(|(_, stats1), (_, stats2)| stats1.score.unwrap().cmp(&stats2.score.unwrap()));

    if maybe_best_split_stats.is_none() {
        //TODO Handle this case
    }

    let (index_of_best_stats, best_split_stats) = maybe_best_split_stats.unwrap();

    let best_split_candidate = split_candidates.get(index_of_best_stats).unwrap();

    println!("Best split {:?}", best_split_stats);

    let (record_ids_left, record_ids_right) = split(
        record_ids_to_split,
        &best_split_candidate,
        &best_split_stats);


}

fn compute_split_stats(
    record_ids_to_split: &Vec<u32>,
    split_candidate: &SplitCandidate,
    labels: &Vec<bool>
) -> SplitStats {
    let mut split_stats = SplitStats::new();

    let attribute_to_split_on = split_candidate.attribute;
    let cut_off = split_candidate.cut_off;

    for record_id in record_ids_to_split {
        // TODO Remove boundary checks here later
        let plus = *labels.get(*record_id as usize).unwrap();
        // TODO We could check here if the values are constant as well
        let is_left = *attribute_to_split_on.get(*record_id as usize).unwrap() < cut_off;

        split_stats.update(plus, is_left);
    }

    split_stats.update_score();

    split_stats
}

fn split(
    record_ids_to_split: Vec<u32>,
    split_candidate: &SplitCandidate,
    split_stats: &SplitStats
) -> (Vec<u32>, Vec<u32>) {

    let attribute_to_split_on = split_candidate.attribute;
    let cut_off = split_candidate.cut_off;

    // TODO We copy here for safety, should reuse the allocated vector later
    let num_left = split_stats.num_plus_left + split_stats.num_minus_left;
    let mut record_ids_left = Vec::with_capacity(num_left as usize);

    let num_right = split_stats.num_plus_right + split_stats.num_minus_right;
    let mut record_ids_right = Vec::with_capacity(num_right as usize);

    for record_id in record_ids_to_split {
        let is_left = *attribute_to_split_on.get(record_id as usize).unwrap() < cut_off;

        if is_left {
            record_ids_left.push(record_id);
        } else {
            record_ids_right.push(record_id);
        }
    }

    (record_ids_left, record_ids_right)
}

