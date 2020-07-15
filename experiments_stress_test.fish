#!/usr/bin/env fish

for i in (seq 0 10);
    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_stress_test
end
