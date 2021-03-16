#!/usr/bin/env fish

for i in (seq 0 20);
    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_forget_leafs
end