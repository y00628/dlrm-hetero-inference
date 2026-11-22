#!/usr/bin/env python3
"""Convert a Meta dlrm_datasets embedding-bag trace (facebookresearch/dlrm_datasets) to our trace format.

    python3 tools/convert_meta.py fbgemm_t856_bs65536_0.pt.gz --name meta --tables 8 --batch-size 2048 --batches 30

Input: a (possibly gzipped) torch file holding a tuple (indices, offsets, lengths): one batch of
`bs` samples for `T` tables (both numbers are in the file name). We assume the FBGEMM layout, where the
bags are ordered table by table: bag (t, b) is bag number t*bs + b, so table t owns
offsets[t*bs .. (t+1)*bs]. The script checks this is consistent with the file (offsets are monotone,
lengths agree with offsets, number of bags is a multiple of bs) and stops if it is not.

What the conversion does:
  - Keeps the first --tables tables.
  - Splits the single batch of `bs` samples into batches of --batch-size (up to --batches of them).
  - Renumbers the row IDs of each table to 0..U-1 (U = distinct IDs used), keeping the exact reuse pattern
    but shrinking the tables. With --max-rows, IDs are folded with a modulo instead.
  - The Meta files have no dense features and no table sizes, so dense features are random
    (seeded) and `rows` in the config is the number of distinct IDs seen.

The files are large (the 2021 batch has ~0.9 billion indices). Loading needs several GB of RAM.
torch.load is called with its default weights_only=True, so the pickle cannot run code.
"""

import argparse
import gzip
import re

import numpy as np

from trace_writer import write_config, write_trace


def load_meta(path):
    import torch

    if path.endswith(".gz"):
        with gzip.open(path) as f:
            obj = torch.load(f)
    else:
        try:
            obj = torch.load(path, mmap=True)
        except Exception:
            obj = torch.load(path)
    indices, offsets, lengths = obj
    return indices.numpy(), offsets.numpy(), (lengths.numpy() if lengths is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="fbgemm_t*_bs*.pt or .pt.gz")
    ap.add_argument("--name", default="meta")
    ap.add_argument("--tables", type=int, default=8, help="keep the first N tables")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--file-batch-size", type=int, help="samples in the file (default: from the file name)")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=0, help="fold IDs modulo this many rows (0 = keep all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="trace path (default data/<name>.trace)")
    ap.add_argument("--config-out", help="config path (default configs/<name>.cfg)")
    args = ap.parse_args()

    bs = args.file_batch_size
    if bs is None:
        m = re.search(r"bs(\d+)", args.input)
        if not m:
            raise SystemExit("cannot read the batch size from the file name; pass --file-batch-size")
        bs = int(m.group(1))

    indices, offsets, lengths = load_meta(args.input)
    nbags = len(offsets) - 1
    print(f"indices {indices.dtype}{indices.shape}, offsets {offsets.dtype}{offsets.shape}, bags {nbags}")

    # Check the layout assumption
    if nbags % bs != 0:
        raise SystemExit(f"{nbags} bags is not a multiple of the batch size {bs}: layout assumption is wrong")
    T = nbags // bs
    diffs = np.diff(offsets)
    if offsets[0] != 0 or offsets[-1] != len(indices) or (diffs < 0).any():
        raise SystemExit("offsets are not a valid CSR offset array for these indices")
    if lengths is not None and lengths.size == nbags and not np.array_equal(lengths.reshape(-1), diffs):
        raise SystemExit("lengths disagree with offsets")
    print(f"{T} tables x {bs} samples; mean pooling per table: "
          f"min {diffs.reshape(T, bs).mean(1).min():.1f}, max {diffs.reshape(T, bs).mean(1).max():.1f}")

    B = args.batch_size
    nb = min(args.batches, bs // B)
    M = min(args.tables, T)
    if nb == 0:
        raise SystemExit(f"batch size {B} is larger than the {bs} samples in the file")
    print(f"using {M} tables, {nb} batches of {B}")

    rng = np.random.default_rng(args.seed)
    dense = rng.random((nb, B, 13), dtype=np.float32)

    out_offsets, out_indices, rows, pooling = [], [], [], []
    for t in range(M):
        first = t * bs  # first bag of table t
        lo, hi = int(offsets[first]), int(offsets[first + nb * B])
        used = indices[lo:hi]
        uniq, ids = np.unique(used, return_inverse=True)
        u = len(uniq)
        if args.max_rows and u > args.max_rows:
            ids = ids % args.max_rows
            u = args.max_rows
        ids = ids.astype(np.int32)
        offs, idxs = [], []
        for k in range(nb):
            b0 = first + k * B
            o = offsets[b0 : b0 + B + 1].astype(np.int64) - offsets[b0]
            s = int(offsets[b0]) - lo
            offs.append(o.astype(np.int32))
            idxs.append(ids[s : s + int(o[-1])])
        out_offsets.append(np.stack(offs))
        out_indices.append(idxs)
        rows.append(u)
        pooling.append(max(1, int(round((hi - lo) / (nb * B)))))

    out = args.out or f"data/{args.name}.trace"
    write_trace(out, dense, out_offsets, out_indices)
    write_config(args.config_out or f"configs/{args.name}.cfg", args.name, rows, pooling, args.emb_dim, B, nb,
                 args.seed, out,
                 f"Meta dlrm_datasets {args.input.split('/')[-1]}, first {M} tables (generated by "
                 f"tools/convert_meta.py); pooling = rounded mean, bag sizes in the trace vary; dense is random")
    print("rows per table:", rows)
    print("mean pooling per table:", pooling)


if __name__ == "__main__":
    main()
