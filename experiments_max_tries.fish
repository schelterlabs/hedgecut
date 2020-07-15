#!/usr/bin/env fish
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_max_tries
