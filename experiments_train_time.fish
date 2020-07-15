#!/usr/bin/env fish

for i in (seq 0 10);

    python3.6 python/train_time.py
    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_train_time

end