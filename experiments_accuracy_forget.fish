#!/usr/bin/env fish

for i in (seq 0 29);
    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_accuracy_forget
end
