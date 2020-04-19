extern crate timely;
extern crate differential_dataflow;
extern crate rand;

use rand::Rng;
use differential_dataflow::input::InputSession;
use differential_dataflow::operators::{Reduce, Join, CountTotal};

use std::error::Error;
use std::str::FromStr;

#[derive(Debug)]
struct Passenger {
    survived: bool,
    class: u8,
    male: bool,
    age: u8,
    siblings: u8,
    parents: u8,
    fare: u32
}


fn titanic_data() -> Result<Vec<Passenger>, Box<dyn Error>> {
    // Build the CSV reader and iterate over each record.
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_path("titanic.csv")?;

    let mut passengers = Vec::with_capacity(887);

    for result in reader.records() {
        let record = result?;

        let survived: bool = u8::from_str(record.get(0).unwrap()).unwrap() == 0;
        let class: u8 = u8::from_str(record.get(1).unwrap()).unwrap();
        let male: bool = record.get(3).unwrap() == "male";
        let age: u8 = f32::from_str(record.get(4).unwrap()).unwrap() as u8;
        let siblings: u8 = u8::from_str(record.get(5).unwrap()).unwrap();
        let parents: u8 = u8::from_str(record.get(6).unwrap()).unwrap();
        let fare: u32 = (f32::from_str(record.get(7).unwrap()).unwrap() * 10000_f32) as u32;

        let passenger = Passenger { survived, class, male, age, siblings, parents, fare };

        passengers.push(passenger);
    }

    Ok(passengers)
}

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let mut observations_input = InputSession::new();

        let probe = worker.dataflow(|scope| {

            let observations = observations_input.to_collection(scope);

            let random_cut = observations
                .map(|(attribute, _label)| ((), attribute))
                .reduce(|_, attribute_recs, output| {

                    // TODO DD should be aware of the semantics here
                    let min_attr = attribute_recs.iter().map(|(attr, _)| **attr).min().unwrap();
                    let max_attr = attribute_recs.iter().map(|(attr, _)| **attr).max().unwrap();

                    let cut_point = rand::thread_rng().gen_range(min_attr, max_attr);

                    output.push((cut_point, 1))
                });

            let counts = observations
                .map(|(attribute, label)| ((), (attribute, label)))
                .join_map(&random_cut, |_, (attribute, label), cut| (*attribute <= *cut, *label))
                .count_total();

            let scores = counts
                .map(|((side, label), count)| ((), (side, label, count)))
                .reduce(|_, side_label_counts, output| {
                    let mut split_stats = [0_u32; 4];

                    for side_label_count in side_label_counts {

                        let (is_left, is_plus, count) = side_label_count.0;

                        let pos =
                            if *is_left {
                                if *is_plus { 0 } else { 1 }
                            } else {
                                if *is_plus { 2 } else { 3 }
                            };

                        split_stats[pos] = *count as u32;
                    }

                    let score = split_score(split_stats[0], split_stats[1], split_stats[2], split_stats[3]);

                    output.push((score, 1));
                });

            scores
                .inspect(|x| println!("\t{:?}", x))
                .probe()
        });

        let passengers = titanic_data().unwrap();
        println!("{} passengers", passengers.len());

        for passenger in passengers {
            observations_input.insert((passenger.age, passenger.survived))
        }

        observations_input.advance_to(0);
        observations_input.flush();

        worker.step_while(|| probe.less_than(observations_input.time()));

    }).unwrap();
}

fn split_score(
    num_plus_left: u32,
    num_minus_left: u32,
    num_plus_right: u32,
    num_minus_right: u32)
-> u32 {

    let num_left = num_plus_left + num_minus_left;
    let num_right = num_plus_right + num_minus_right;
    let num_plus = num_plus_left + num_plus_right;
    let num_minus = num_minus_left + num_minus_right;

    let num_samples = num_left + num_right;

    // Prior "classification entropy" H_C(S)
    let hcs = H(num_plus, num_minus, num_samples);

    // Entropy of S with respect to test T H_T(S)
    let hts = H(num_left, num_right, num_samples);

    // Posterior "classification entropy" H_{C|T}(S) of S given the outcome of the test T
    // TODO this is computed twice
    let p_sys = num_left as f32 / num_samples as f32;
    let p_sns = num_right as f32 / num_samples as f32;

    let hcsy = H(num_plus_left, num_minus_left, num_left);
    let hcsn = H(num_plus_right, num_minus_right, num_right);

    let hcts = p_sys * hcsy + p_sns * hcsn;

    // Information gain of applying test T
    let icts = hcs - hcts;

    let score = 2.0 * icts / (hcs + hts);

    // println!("{}, {}, cut {}, H_C(S) {}, H_T(s) {}, H_C|T(S) {}", min_attribute, max_attribute,
    //          cut_point, hcs, hts, hcts);

    // println!("{}, {}", cut_point, score);

    (score * 1_000_000_f32) as u32
}

#[allow(non_snake_case)]
fn H(a: u32, b: u32, a_plus_b: u32) -> f32 {
    let p_a = a as f32 / a_plus_b as f32;
    let p_b = b as f32 / a_plus_b as f32;
    -(p_a * p_a.log2() + p_b * p_b.log2())
}
