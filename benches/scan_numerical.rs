#[macro_use]
extern crate bencher;
extern crate hedgecut;

use bencher::Bencher;

use hedgecut::dataset::GiveMeSomeCreditDataset;
use hedgecut::tree::Split;

benchmark_group!(benches, bench_scan_with_branches, bench_scan_mlpack, bench_scan, bench_scan_simd);
benchmark_main!(benches);

fn bench_scan_with_branches(bench: &mut Bencher) {

    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let split = Split::Numerical { attribute_index: 3, cut_off: 11 };

    bench.iter(|| {
        bencher::black_box(hedgecut::scan::scan_with_branches(&samples, &split));
    })
}

fn bench_scan_mlpack(bench: &mut Bencher) {

    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let split = Split::Numerical { attribute_index: 3, cut_off: 11 };

    bench.iter(|| {
        bencher::black_box(hedgecut::scan::scan_mlpack(&samples, &split));
    })
}

fn bench_scan(bench: &mut Bencher) {

    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let split = Split::Numerical { attribute_index: 3, cut_off: 11 };

    bench.iter(|| {
        bencher::black_box(hedgecut::scan::scan(&samples, &split));
    })
}

fn bench_scan_simd(bench: &mut Bencher) {

    let samples = GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");
    let split = Split::Numerical { attribute_index: 3, cut_off: 11 };

    bench.iter(|| {
        bencher::black_box(hedgecut::scan::scan_simd_numerical(&samples, &split));
    })
}
