#!/usr/bin/env fish

for i in (seq 0 10);

    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_robustness_accuracy

end