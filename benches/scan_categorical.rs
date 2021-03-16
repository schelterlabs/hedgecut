#[macro_use]
extern crate bencher;
extern crate hedgecut;

use bencher::Bencher;

use hedgecut::dataset::ShoppingDataset;
use hedgecut::tree::Split;

benchmark_group!(benches, bench_scan_with_branches, bench_scan_mlpack, bench_scan, bench_scan_simd);
benchmark_main!(benches);


fn bench_scan_with_branches(bench: &mut Bencher) {

    let samples = ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");
    let possible_values = vec![0, 7, 12];

    let mut subset: u64 = 0;
    for bit_to_set in possible_values.iter() {
        subset |= 1_u64 << *bit_to_set as u64
    }

    let split = Split::Categorical { attribute_index: 14, subset };

    bench.iter(|| {
        bencher::black_box(hedgecut::scan::scan_with_branches(&samples, &split));
    })
}

fn bench_scan_mlpack(bench: &mut Bencher) {

    let samples = ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");
    let possible_values = vec![0, 7, 12];

    let mut subset: u64 = 0;
    for bit_to_set in possible_values.iter() {
        subset |= 1_u64 << *bit_to_set as u64
    }

    let split = Split::Categorical { attribute_index: 14, subset };

    bench.iter(|| {
        bencher::black_box(hedgecut::scan::scan_mlpack(&samples, &split));
    })
}

fn bench_scan(bench: &mut Bencher) {

    let samples = ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");
    let possible_values = vec![0, 7, 12];

    let mut subset: u64 = 0;
    for bit_to_set in possible_values.iter() {
        subset |= 1_u64 << *bit_to_set as u64
    }

    let split = Split::Categorical { attribute_index: 14, subset };

    bench.iter(|| {
        bencher::black_box(hedgecut::scan::scan(&samples, &split));
    })
}

fn bench_scan_simd(bench: &mut Bencher) {

    let samples = ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");
    let possible_values = vec![0, 7, 12];

    let mut subset: u64 = 0;
    for bit_to_set in possible_values.iter() {
        subset |= 1_u64 << *bit_to_set as u64
    }

    let split = Split::Categorical { attribute_index: 14, subset };

    bench.iter(|| {
        bencher::black_box(hedgecut::scan::scan_simd_categorical(&samples, &split));
    })
}
