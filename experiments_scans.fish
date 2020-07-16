#!/usr/bin/env fish

#RUSTFLAGS="-C target-cpu=native" cargo bench --bench scans
RUSTFLAGS="-C target-cpu=native" cargo bench --bench scan_categorical
