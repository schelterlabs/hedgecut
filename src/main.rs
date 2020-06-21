#[macro_use]
extern crate ndarray;
extern crate rand;

use ndarray::prelude::*;
use rand::Rng;

mod titanic;
mod split_scoring;

use titanic::Passenger;
use split_scoring::split_score;

fn split<F>(samples: &Vec<Passenger>, extract_func: F)
    where F: Fn(&Passenger) -> (u8, bool)
{

    let attribute_and_label: Vec<_> = samples.iter()
        .map(|sample| extract_func(sample))
        .collect();

    // TODO Can be collected in single pass
    let min_attribute = attribute_and_label.iter().map(|(attribute, _)| attribute).min().unwrap();
    let max_attribute = attribute_and_label.iter().map(|(attribute, _)| attribute).max().unwrap();

    let mut rng = rand::thread_rng();

    let cut_point = rng.gen_range(min_attribute, max_attribute);

    let mut num_plus_left = 0_u32;
    let mut num_minus_left = 0_u32;
    let mut num_plus_right = 0_u32;
    let mut num_minus_right = 0_u32;

    // TODO can be simplified maybe
    attribute_and_label.iter().for_each(|(age, class)| {
        if *age <= cut_point {
            if *class {
                num_plus_left += 1;
            } else {
                num_minus_left += 1;
            }
        } else {
            if *class {
                num_plus_right += 1;
            } else {
                num_minus_right += 1;
            }
        }
    });

    let score = split_score(num_plus_left, num_minus_left, num_minus_right, num_minus_right);

    println!("{}", score);
}


fn main() {

    let passengers = titanic::titanic_data().unwrap();

    println!("{} passengers", passengers.len());

    //let extract_age = |passenger: &Passenger| (passenger.age, passenger.survived);

    let mut ages: Vec<(u32, u8)> = passengers.iter()
        .map(|passenger| (passenger.index, passenger.age))
        .collect();

    ages.sort_unstable_by_key(|r| r.1);

    let step_size = ages.len() / 16;

    println!("{}", step_size);


    //split(&passengers, extract_age);
     // M number of trees in ensemble, n_min minimal number of samples per leaf,
     // K number of attributes to test for splits

}