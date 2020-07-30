extern crate hedgecut;

use hedgecut::dataset::{AdultDataset, CardioDataset, GiveMeSomeCreditDataset, PropublicaDataset, ShoppingDataset};
use hedgecut::evaluation::max_tries;

fn main() {
    let num_trees = 100;
    let min_leaf_size = 2;


    let samples = AdultDataset::samples_from_csv("datasets/adult-train.csv");
    let test_data = AdultDataset::samples_from_csv("datasets/adult-test.csv");
    let dataset = AdultDataset::from_samples(&samples);
    let max_tries_per_split = vec![1, 3, 5, 10, 25, 50, 100, 150, 200, 250];

    max_tries(
        "adult",
        dataset,
        samples,
        test_data,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );


    let samples = CardioDataset::samples_from_csv("datasets/cardio-train.csv");
    let test_data = CardioDataset::samples_from_csv("datasets/cardio-test.csv");
    let dataset = CardioDataset::from_samples(&samples);
    let max_tries_per_split = vec![1, 3, 5, 10, 25, 50, 100, 150, 200, 250];

    max_tries(
        "cardio",
        dataset,
        samples,
        test_data,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );


    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let test_data = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-test.csv");
    let dataset = GiveMeSomeCreditDataset::from_samples(&samples);
    let max_tries_per_split = vec![1, 3, 5, 10, 25, 50, 100, 150, 200, 250];

    max_tries(
        "givemesomecredit",
        dataset,
        samples,
        test_data,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );


    let samples = PropublicaDataset::samples_from_csv("datasets/propublica-train.csv");
    let test_data = PropublicaDataset::samples_from_csv("datasets/propublica-test.csv");
    let dataset = PropublicaDataset::from_samples(&samples);
    let max_tries_per_split = vec![1, 3, 5, 10, 25, 50, 100, 150, 200, 250];

    max_tries(
        "propublica",
        dataset,
        samples,
        test_data,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );


    let samples = ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");
    let test_data = ShoppingDataset::samples_from_csv("datasets/shopping-test.csv");
    let dataset = ShoppingDataset::from_samples(&samples);
    let max_tries_per_split = vec![1, 3, 5, 10, 25, 50, 100, 150, 200, 250];

    max_tries(
        "shopping",
        dataset,
        samples,
        test_data,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );
}