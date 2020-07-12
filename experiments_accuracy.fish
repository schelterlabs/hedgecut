#!/usr/bin/env fish

for i in (seq 0 10);

    python3.6 python/prepare_adult.py
    python3.6 python/sklearn_adult.py
    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_on_adult

    python3.6 python/prepare_cardio.py
    python3.6 python/sklearn_cardio.py
    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_on_cardio

    python3.6 python/prepare_givemesomecredit.py
    python3.6 python/sklearn_givemesomecredit.py
    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_on_givemesomecredit

    python3.6 python/prepare_propublica.py
    python3.6 python/sklearn_propublica.py
    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_on_propublica

    python3.6 python/prepare_shopping.py
    python3.6 python/sklearn_shopping.py
    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin evaluate_on_shopping

end