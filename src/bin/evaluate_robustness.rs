extern crate hedgecut;

use hedgecut::evaluation::robustness;
use hedgecut::dataset::{AdultDataset, CardioDataset, GiveMeSomeCreditDataset, PropublicaDataset, ShoppingDataset};

fn main() {
    let num_trees = 100;
    let min_leaf_size = 2;


    let samples = AdultDataset::samples_from_csv("datasets/adult-train.csv");
    let dataset = AdultDataset::from_samples(&samples);
    let max_tries_per_split = 5;

    robustness(
        "adult",
        dataset,
        samples,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );


    let samples = CardioDataset::samples_from_csv("datasets/cardio-train.csv");
    let dataset = CardioDataset::from_samples(&samples);
    let max_tries_per_split = 5;

    robustness(
        "cardio",
        dataset,
        samples,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );


    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let dataset = GiveMeSomeCreditDataset::from_samples(&samples);
    let max_tries_per_split = 5;

    robustness(
        "givemesomecredit",
        dataset,
        samples,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );

    let samples = PropublicaDataset::samples_from_csv("datasets/propublica-train.csv");
    let dataset = PropublicaDataset::from_samples(&samples);
    let max_tries_per_split = 5;

    robustness(
        "propublica",
        dataset,
        samples,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );

    let samples = ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");
    let dataset = ShoppingDataset::from_samples(&samples);
    let max_tries_per_split = 50;

    robustness(
        "shopping",
        dataset,
        samples,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );
}