extern crate timely;
extern crate differential_dataflow;
extern crate rand;

use rand::Rng;
use differential_dataflow::input::InputSession;
use differential_dataflow::operators::{Reduce, Join, CountTotal};

mod titanic;
mod split_scoring;

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

                    for ((is_left, is_plus, count), _) in side_label_counts {
                        let pos =
                            if *is_left {
                                if *is_plus { 0 } else { 1 }
                            } else {
                                if *is_plus { 2 } else { 3 }
                            };

                        split_stats[pos] = *count as u32;
                    }

                    let score = split_scoring::split_score(
                        split_stats[0],
                        split_stats[1],
                        split_stats[2],
                        split_stats[3]
                    );

                    output.push((score, 1));
                });

            scores
                .inspect(|x| println!("\t{:?}", x))
                .probe()
        });

        let passengers = titanic::titanic_data().unwrap();
        println!("{} passengers", passengers.len());

        for passenger in passengers {
            observations_input.insert((passenger.age, passenger.survived))
        }

        observations_input.advance_to(0);
        observations_input.flush();

        worker.step_while(|| probe.less_than(observations_input.time()));

    }).unwrap();
}
