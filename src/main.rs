extern crate csv;
extern crate rand;
extern crate rayon;

mod split_stats;
mod dataset;
mod tree;



use tree::ExtremelyRandomizedTrees;
use dataset::{TitanicDataset, TitanicSample};




fn main() {

    let dataset = TitanicDataset::from_csv();

    let trees = ExtremelyRandomizedTrees::fit(&dataset, 100);

    let sample = TitanicSample {
        age: 5,
        fare: 1,
        siblings: 1,
        children: 0,
        gender: 1,
        pclass: 3,
        label: false
    };

    let predicted_label = trees.predict(&sample);

    println!("Predicted label: {}, True label: {}", predicted_label, sample.label);

}
