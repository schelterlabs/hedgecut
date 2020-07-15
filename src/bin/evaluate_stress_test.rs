extern crate hedgecut;

use hedgecut::dataset::{AdultDataset, CardioDataset, GiveMeSomeCreditDataset, PropublicaDataset, ShoppingDataset};
use hedgecut::evaluation::stress_test;

fn main() {
    let num_trees = 100;
    let min_leaf_size = 2;


    let samples = AdultDataset::samples_from_csv("datasets/adult-train.csv");
    let test_data = AdultDataset::samples_from_csv("datasets/adult-test.csv");
    let dataset = AdultDataset::from_samples(&samples);

    stress_test(
        "adult",
        dataset,
        samples,
        test_data,
        16,
        num_trees,
        min_leaf_size,
        50
    );


    let samples = CardioDataset::samples_from_csv("datasets/cardio-train.csv");
    let test_data = CardioDataset::samples_from_csv("datasets/cardio-test.csv");
    let dataset = CardioDataset::from_samples(&samples);

    stress_test(
        "cardio",
        dataset,
        samples,
        test_data,
        7,
        num_trees,
        min_leaf_size,
        50
    );


    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let test_data = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-test.csv");
    let dataset = GiveMeSomeCreditDataset::from_samples(&samples);

    stress_test(
        "givemesomecredit",
        dataset,
        samples,
        test_data,
        4,
        num_trees,
        min_leaf_size,
        50
    );


    let samples = PropublicaDataset::samples_from_csv("datasets/propublica-train.csv");
    let test_data = PropublicaDataset::samples_from_csv("datasets/propublica-test.csv");
    let dataset = PropublicaDataset::from_samples(&samples);

    stress_test(
        "propublica",
        dataset,
        samples,
        test_data,
        75,
        num_trees,
        min_leaf_size,
        50
    );


    let samples = ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");
    let test_data = ShoppingDataset::samples_from_csv("datasets/shopping-test.csv");
    let dataset = ShoppingDataset::from_samples(&samples);

    stress_test(
        "shopping",
        dataset,
        samples,
        test_data,
        40,
        num_trees,
        min_leaf_size,
        250
    );
}