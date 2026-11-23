#!/usr/bin/env python3
"""Generate a synthetic Zipf trace in the format described in include/trace.h.

Run from the repo root:
    python3 tools/gen_trace.py configs/small.cfg
    # separate trace for choosing hot rows (different seed)
    python3 tools/gen_trace.py configs/small.cfg --seed 1001 --out data/small_profile.trace

Row popularity follows Zipf(alpha), sampled from a precomputed CDF (works for alpha <= 1,
unlike numpy.random.zipf). Popularity ranks are mapped to row IDs with (a * rank + c) mod N,
so the hottest rows are spread over the table instead of being rows 0, 1, 2, ...
"""

import argparse
import math
import os

import numpy as np


def load_config(path):
    cfg = {}
    for line in open(path):
        line = line.split("#")[0]
        if "=" in line:
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip()
    return cfg


def per_table(value, num_tables):
    vals = [int(v) for v in value.split(",")]
    return vals * num_tables if len(vals) == 1 else vals


def sample_rows(rng, n, alpha, count, table):
    # popularity rank in [0, n), rank 0 is the hottest
    if alpha == 0:
        ranks = rng.integers(0, n, size=count)
    else:
        cdf = np.cumsum(np.arange(1, n + 1, dtype=np.float64) ** (-alpha))
        cdf /= cdf[-1]
        ranks = np.minimum(np.searchsorted(cdf, rng.random(count)), n - 1)

    # rank -> row ID with a fixed bijection per table. It must not depend on the seed:
    # a profiling trace (different seed) has to find the same hot rows as the main trace.
    a = 2654435761 % n
    while math.gcd(a, n) != 1:
        a += 1
    c = (table * 104729 + 12345) % n
    rows = (ranks.astype(np.uint64) * np.uint64(a) + np.uint64(c)) % np.uint64(n)
    return rows.astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()

    cfg = load_config(args.config)
    M = int(cfg["num_tables"])
    B = int(cfg["batch_size"])
    num_batches = int(cfg["num_batches"])
    num_dense = int(cfg.get("num_dense", 13))
    alpha = float(cfg["zipf_alpha"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    rows = per_table(cfg["rows"], M)
    pooling = per_table(cfg["pooling"], M)
    out_path = args.out or cfg["trace_file"]

    rng = np.random.default_rng(seed)
    indices = []  # indices[t] has shape [num_batches, B * L]
    for t in range(M):
        idx = sample_rows(rng, rows[t], alpha, num_batches * B * pooling[t], t)
        indices.append(idx.reshape(num_batches, B * pooling[t]))
    dense = rng.random((num_batches, B, num_dense), dtype=np.float32)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(np.array([M, B, num_batches, num_dense], dtype=np.int32).tobytes())
        for b in range(num_batches):
            f.write(dense[b].tobytes())
            for t in range(M):
                offsets = np.arange(B + 1, dtype=np.int32) * pooling[t]
                f.write(offsets.tobytes())
                f.write(indices[t][b].tobytes())

    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
