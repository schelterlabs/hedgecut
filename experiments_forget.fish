#!/usr/bin/env fish

for i in (seq 0 3);

    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_forget
    python3.6 python/forget.py

end