"""Shared helpers for the data converters: write a trace (include/trace.h) and a matching config.

Trace layout (little-endian):
    header:    int32 num_tables, batch_size, num_batches, num_dense
    per batch: float32 dense[B][num_dense]
               for each table: int32 offsets[B+1], int32 indices[offsets[B]]
"""

import os

import numpy as np


def write_trace(path, dense, offsets, indices):
    """dense: float32 [nb, B, nd]; offsets: per table, int32 [nb, B+1]; indices: per table, list of nb int32 arrays."""
    nb, B, nd = dense.shape
    M = len(offsets)
    assert len(indices) == M
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.array([M, B, nb, nd], dtype=np.int32).tobytes())
        for k in range(nb):
            f.write(np.ascontiguousarray(dense[k], dtype=np.float32).tobytes())
            for t in range(M):
                off = np.ascontiguousarray(offsets[t][k], dtype=np.int32)
                idx = np.ascontiguousarray(indices[t][k], dtype=np.int32)
                assert off.shape == (B + 1,) and off[0] == 0 and off[B] == len(idx), (t, k)
                f.write(off.tobytes())
                f.write(idx.tobytes())
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def default_mlps(num_tables, emb_dim, num_dense=13):
    n = num_tables + 1
    z = emb_dim + n * (n - 1) // 2
    return [num_dense, 512, 256, emb_dim], [z, 512, 256, 1]


def write_config(path, name, rows, pooling, emb_dim, batch_size, num_batches, seed, trace_file, comment,
                 num_dense=13):
    """Config in the format of configs/*.cfg. zipf_alpha is 0 because the access pattern is real, not synthetic."""
    M = len(rows)
    bot, top = default_mlps(M, emb_dim, num_dense)
    n = M + 1
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {comment}\n")
        f.write(f"num_tables  = {M}\n")
        f.write(f"rows        = {','.join(str(int(r)) for r in rows)}\n")
        f.write(f"pooling     = {','.join(str(int(p)) for p in pooling)}\n")
        f.write(f"emb_dim     = {emb_dim}\n")
        f.write(f"batch_size  = {batch_size}\n")
        f.write(f"num_batches = {num_batches}\n")
        f.write(f"num_dense   = {num_dense}\n")
        f.write("zipf_alpha  = 0     # real data, not synthetic\n")
        f.write(f"seed        = {seed}\n")
        f.write(f"mlp_bot     = {'-'.join(str(x) for x in bot)}\n")
        f.write(f"mlp_top     = {'-'.join(str(x) for x in top)}     # {emb_dim} + {n}*{n - 1}/2\n")
        f.write(f"trace_file  = {trace_file}\n")
    table_gb = sum(rows) * emb_dim * 4 / 1e9
    print(f"wrote {path} (M={M}, embedding tables {table_gb:.2f} GB)")
