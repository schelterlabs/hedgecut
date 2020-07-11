#!/usr/bin/env fish
RUSTFLAGS="-C target-cpu=native -C link-args=-Wl,-zstack-size=18194304" cargo run --release --bin evaluate_on_defaults


