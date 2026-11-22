#!/usr/bin/env python3
"""Convert the Criteo Kaggle Display Advertising data (train.txt) to our trace format.

    python3 tools/convert_criteo.py /path/to/train.txt --name criteo --batch-size 2048 --batches 60

Input: tab-separated lines  label, I1..I13 (integers, may be empty), C1..C26 (hex strings, may be empty).
Output: data/<name>.trace and configs/<name>.cfg (rows per table are filled in from the data).

What the conversion does:
  - dense: x -> log(1 + max(x, 0)), empty -> 0 (standard DLRM preprocessing). The label is not used.
  - sparse: each of the 26 columns is one table with exactly one ID per sample (pooling factor L = 1).
    IDs are renumbered 0..U-1 per column, where U is the number of distinct values seen in the lines read
    (empty values count as one more value). With --max-rows, IDs are folded with a modulo instead.
  - Only the first batch_size * batches lines are read, so rows per table are small compared to the full set.
"""

import argparse
import os

import numpy as np

from trace_writer import write_config, write_trace

NUM_DENSE = 13
NUM_CAT = 26


def read_parquet(path, count):
    """Hugging Face criteo/CriteoClickLogs layout: columns label, integer_feature_1..13 (int32, nullable),
    categorical_feature_1..26 (8-character hex strings, nullable). Reads only the first `count` rows."""
    import pyarrow.parquet as pq

    cols = [f"integer_feature_{i + 1}" for i in range(NUM_DENSE)] + [f"categorical_feature_{j + 1}" for j in range(NUM_CAT)]
    dense = np.zeros((count, NUM_DENSE), dtype=np.float32)
    cats = np.full((count, NUM_CAT), -1, dtype=np.int64)  # -1 = missing
    n = 0
    for rb in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=cols):
        take = min(rb.num_rows, count - n)
        if take <= 0:
            break
        for i in range(NUM_DENSE):
            col = rb.column(i).slice(0, take).to_pylist()
            dense[n : n + take, i] = [0.0 if v is None else float(v) for v in col]
        for j in range(NUM_CAT):
            col = rb.column(NUM_DENSE + j).slice(0, take).to_pylist()
            cats[n : n + take, j] = [-1 if v is None or v == "" else int(v, 16) for v in col]
        n += take
    return dense[:n], cats[:n]


def read_lines(path, count):
    if path.endswith(".parquet"):
        return read_parquet(path, count)
    dense = np.zeros((count, NUM_DENSE), dtype=np.float32)
    cats = np.full((count, NUM_CAT), -1, dtype=np.int64)  # -1 = missing
    n = 0
    with open(path, "r") as f:
        for line in f:
            if n == count:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 1 + NUM_DENSE + NUM_CAT:
                fields += [""] * (1 + NUM_DENSE + NUM_CAT - len(fields))
            for i in range(NUM_DENSE):
                s = fields[1 + i]
                dense[n, i] = float(s) if s else 0.0
            for j in range(NUM_CAT):
                s = fields[1 + NUM_DENSE + j]
                if s:
                    cats[n, j] = int(s, 16)
            n += 1
    return dense[:n], cats[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Criteo train.txt (tab-separated) or one .parquet part file from Hugging Face "
                                  "criteo/CriteoClickLogs (needs pyarrow)")
    ap.add_argument("--name", default="criteo")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=0, help="fold IDs modulo this many rows (0 = keep all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="trace path (default data/<name>.trace)")
    ap.add_argument("--config-out", help="config path (default configs/<name>.cfg)")
    args = ap.parse_args()

    B = args.batch_size
    dense, cats = read_lines(args.input, B * args.batches)
    nb = len(dense) // B
    if nb == 0:
        raise SystemExit(f"need at least {B} lines, found {len(dense)}")
    if nb < args.batches:
        print(f"warning: only {len(dense)} lines, using {nb} batches instead of {args.batches}")
    dense = np.log1p(np.maximum(dense[: nb * B], 0.0)).reshape(nb, B, NUM_DENSE)
    cats = cats[: nb * B]

    offsets, indices, rows = [], [], []
    for j in range(NUM_CAT):
        uniq, ids = np.unique(cats[:, j], return_inverse=True)
        u = len(uniq)
        if args.max_rows and u > args.max_rows:
            ids = ids % args.max_rows
            u = args.max_rows
        ids = ids.astype(np.int32).reshape(nb, B)
        offsets.append(np.tile(np.arange(B + 1, dtype=np.int32), (nb, 1)))  # one ID per sample
        indices.append([ids[k] for k in range(nb)])
        rows.append(u)

    out = args.out or f"data/{args.name}.trace"
    write_trace(out, dense, offsets, indices)
    write_config(args.config_out or f"configs/{args.name}.cfg", args.name, rows, [1] * NUM_CAT, args.emb_dim, B,
                 nb, args.seed, out,
                 f"Criteo Kaggle, first {nb * B} lines, one ID per feature (generated by tools/convert_criteo.py)")
    print("rows per table:", rows)


if __name__ == "__main__":
    main()
