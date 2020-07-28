#!/usr/bin/env fish

RUSTFLAGS="-C target-cpu=native" cargo bench --bench scan_numerical
RUSTFLAGS="-C target-cpu=native" cargo bench --bench scan_categorical
