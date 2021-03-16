extern crate hedgecut;

use hedgecut::dataset::{AdultDataset, CardioDataset, GiveMeSomeCreditDataset, PropublicaDataset, ShoppingDataset};
use hedgecut::evaluation::forget2;

fn main() {
    let num_trees = 100;

    for min_leaf_size in &[2, 4, 8, 16, 32, 64, 128] {

        let samples = AdultDataset::samples_from_csv("datasets/adult-train.csv");
        let dataset = AdultDataset::from_samples(&samples);

        forget2(
            "adult",
            dataset,
            samples,
            num_trees,
            *min_leaf_size,
            5
        );

        let samples = CardioDataset::samples_from_csv("datasets/cardio-train.csv");
        let dataset = CardioDataset::from_samples(&samples);

        forget2(
            "cardio",
            dataset,
            samples,
            num_trees,
            *min_leaf_size,
            5
        );

        let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
        let dataset = GiveMeSomeCreditDataset::from_samples(&samples);

        forget2(
            "givemesomecredit",
            dataset,
            samples,
            num_trees,
            *min_leaf_size,
            5
        );

        let samples = PropublicaDataset::samples_from_csv("datasets/propublica-train.csv");
        let dataset = PropublicaDataset::from_samples(&samples);

        forget2(
            "propublica",
            dataset,
            samples,
            num_trees,
            *min_leaf_size,
            5
        );

        let samples = ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");
        let dataset = ShoppingDataset::from_samples(&samples);

        forget2(
            "shopping",
            dataset,
            samples,
            num_trees,
            *min_leaf_size,
            50
        );
    }



/*

    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let dataset = GiveMeSomeCreditDataset::from_samples(&samples);
    let max_tries_per_split = 5;

    forget(
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

    forget(
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

    forget(
        "shopping",
        dataset,
        samples,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );*/
}