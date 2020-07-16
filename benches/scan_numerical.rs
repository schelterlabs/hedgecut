#[macro_use]
extern crate bencher;
extern crate hedgecut;

use bencher::Bencher;

use hedgecut::dataset::{GiveMeSomeCreditDataset, Dataset};
use rand::{XorShiftRng, SeedableRng, Rng};
use hedgecut::tree::Split;

benchmark_group!(benches, bench_scan, bench_scan_simd);

benchmark_main!(benches);

const NUM_REPETITIONS: usize = 3;
const NUM_SPLITS: usize = 20;

fn bench_scan(bench: &mut Bencher) {

    let mut rng =
        XorShiftRng::from_seed([0, 1, 2, 3, 4 , 5 , 6, 7, 8, 9, 10, 11, 12 , 13 , 14, 15]);

    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let dataset = GiveMeSomeCreditDataset::from_samples(&samples);

    let splits: Vec<Split> = (0..NUM_SPLITS)
        .map(|_| {
            let attribute_index = rng.gen_range(0, dataset.num_attributes());
            let (_, max_value) = dataset.attribute_range(attribute_index);
            let cut_off = rng.gen_range(0, max_value);

            Split::Numerical { attribute_index, cut_off }
        })
        .collect();

    bench.iter(|| {
        for _ in 0..NUM_REPETITIONS {
            for split in &splits {
                bencher::black_box(hedgecut::scan::scan(&samples, split));
            }
        }
    })
}

fn bench_scan_simd(bench: &mut Bencher) {

    let mut rng =
        XorShiftRng::from_seed([0, 1, 2, 3, 4 , 5 , 6, 7, 8, 9, 10, 11, 12 , 13 , 14, 15]);

    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let dataset = GiveMeSomeCreditDataset::from_samples(&samples);

    let splits: Vec<Split> = (0..NUM_SPLITS)
        .map(|_| {
            let attribute_index = rng.gen_range(0, dataset.num_attributes());
            let (_, max_value) = dataset.attribute_range(attribute_index);
            let cut_off = rng.gen_range(0, max_value);

            Split::Numerical { attribute_index, cut_off }
        })
        .collect();

    bench.iter(|| {
        for _ in 0..NUM_REPETITIONS {
            for split in &splits {
                bencher::black_box(hedgecut::scan::scan_simd_numerical(&samples, split));
            }
        }
    })
}
