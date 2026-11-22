#!/usr/bin/env python3
"""Convert the MLPerf DLRM v2 multi-hot Criteo dataset to our trace format.

    python3 tools/convert_mlperf.py /path/to/dir --suffix _debug --name mlperf --tables 8 --batch-size 2048 --batches 30

Input directory holds day_23_dense<suffix>.npy and day_23_sparse_multi_hot<suffix>.npz
(suffix is "" for the full set, "_sample" or "_debug" for the small ones).

Format as documented by MLCommons (not checked against the real files):
  - sparse: an .npz with keys "0".."25" (one per table), each an int32 array [N][width_j]. The width is fixed per
    table, so bag b of a table is row b, and offsets are arange(B+1) * width.
  - dense:  an .npy [N][13]. We use it as it is (--dense-transform log applies log(1 + max(x, 0)) if the file turns out
    to hold raw counts). A summary of its range is printed so you can tell.

What the conversion does:
  - Reads only the first batch_size * batches rows. The npz members are memory-mapped, so a multi-GB file is not loaded.
  - Keeps the first --tables tables, or the ones named by --table-ids (e.g. 20,21 are the widest bags).
  - Renumbers the row IDs of each table to 0..U-1 (U = distinct IDs used), keeping the exact reuse pattern.
    With --max-rows, IDs are folded with a modulo instead.
  - MLPerf uses embedding dim 128, so --emb-dim defaults to 128.
Note: the extra IDs in each bag are synthetic and uniformly random (only the first ID of a bag is the real Criteo ID).
"""

import argparse
import os
import struct
import zipfile

import numpy as np

from trace_writer import write_config, write_trace


def open_npz_member(path, key):
    """Memory-map one array of an uncompressed .npz. Falls back to loading it fully if it is compressed."""
    name = f"{key}.npy"
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name)
        compressed = info.compress_type != zipfile.ZIP_STORED
    if compressed:
        print(f"warning: {name} in {path} is compressed, loading it fully into memory")
        with np.load(path) as z:
            return z[key]
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)  # local file header: name and extra lengths can differ from the central directory
        name_len, extra_len = struct.unpack("<HH", local[26:30])
        f.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        data_offset = f.tell()
    if fortran:
        raise SystemExit(f"{name} is stored in Fortran order, which this converter does not handle")
    return np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory with the day_23 files")
    ap.add_argument("--suffix", default="", help='"", "_sample" or "_debug"')
    ap.add_argument("--day", type=int, default=23)
    ap.add_argument("--name", default="mlperf")
    ap.add_argument("--tables", type=int, default=8, help="keep the first N tables")
    ap.add_argument("--table-ids", help="comma-separated table numbers to keep (overrides --tables)")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--max-rows", type=int, default=0, help="fold IDs modulo this many rows (0 = keep all)")
    ap.add_argument("--dense-transform", choices=["none", "log"], default="none")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="trace path (default data/<name>.trace)")
    ap.add_argument("--config-out", help="config path (default configs/<name>.cfg)")
    args = ap.parse_args()

    dense_path = os.path.join(args.dir, f"day_{args.day}_dense{args.suffix}.npy")
    sparse_path = os.path.join(args.dir, f"day_{args.day}_sparse_multi_hot{args.suffix}.npz")
    dense_all = np.load(dense_path, mmap_mode="r")
    with zipfile.ZipFile(sparse_path) as z:
        keys = sorted((n[:-4] for n in z.namelist() if n.endswith(".npy")), key=int)
    print(f"{len(keys)} tables, dense {dense_all.dtype}{dense_all.shape}")

    B = args.batch_size
    avail = dense_all.shape[0]
    nb = min(args.batches, avail // B)
    if nb == 0:
        raise SystemExit(f"need at least {B} rows, the file has {avail}")
    n = nb * B

    table_ids = [int(x) for x in args.table_ids.split(",")] if args.table_ids else list(range(min(args.tables, len(keys))))
    for t in table_ids:
        if t < 0 or t >= len(keys):
            raise SystemExit(f"table {t} does not exist (file has {len(keys)})")
    print(f"using tables {table_ids}, {nb} batches of {B}")

    dense = np.array(dense_all[:n], dtype=np.float32)
    nd = dense.shape[1]
    print(f"dense: {nd} columns, min {dense.min():.3g}, max {dense.max():.3g}, mean {dense.mean():.3g}")
    if not np.isfinite(dense).all():
        raise SystemExit("dense features contain NaN or inf")
    if args.dense_transform == "log":
        dense = np.log1p(np.maximum(dense, 0.0))
    dense = dense.reshape(nb, B, nd)

    offsets, indices, rows, pooling = [], [], [], []
    for t in table_ids:
        arr = open_npz_member(sparse_path, keys[t])
        if arr.ndim != 2 or arr.shape[0] < n:
            raise SystemExit(f"table {t}: expected a 2-D array with at least {n} rows, got shape {arr.shape}")
        if arr.dtype != np.int32:
            raise SystemExit(f"table {t}: expected int32, got {arr.dtype}")
        w = arr.shape[1]
        used = np.asarray(arr[:n]).reshape(-1)
        if used.min() < 0:
            raise SystemExit(f"table {t}: negative IDs")
        uniq, ids = np.unique(used, return_inverse=True)
        u = len(uniq)
        if args.max_rows and u > args.max_rows:
            ids = ids % args.max_rows
            u = args.max_rows
        ids = ids.astype(np.int32).reshape(nb, B * w)
        offsets.append(np.tile(np.arange(B + 1, dtype=np.int32) * w, (nb, 1)))
        indices.append([ids[k] for k in range(nb)])
        rows.append(u)
        pooling.append(w)

    out = args.out or f"data/{args.name}.trace"
    write_trace(out, dense, offsets, indices)
    write_config(args.config_out or f"configs/{args.name}.cfg", args.name, rows, pooling, args.emb_dim, B, nb,
                 args.seed, out,
                 f"MLPerf DLRM v2 multi-hot Criteo day {args.day}{args.suffix}, tables {table_ids}, first {n} rows "
                 f"(generated by tools/convert_mlperf.py); extra IDs in each bag are synthetic",
                 num_dense=nd)
    print("rows per table:", rows)
    print("pooling (bag width) per table:", pooling)


if __name__ == "__main__":
    main()
