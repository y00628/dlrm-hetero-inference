#!/usr/bin/env python3
"""Hot-row profiler: reads a profiling trace and outputs (table, row) pairs sorted by frequency.

Format of hot_file:
    Repeated little-endian int32 pairs: [table_id, row_id]
    Sorted from highest frequency to lowest frequency.
"""

import argparse
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate hot-row list from a profiling trace.")
    parser.add_argument("trace", help="Path to profiling trace file (e.g. data/small_profile.trace)")
    parser.add_argument("--out", required=True, help="Output hot list binary (e.g. data/small_hot.bin)")
    args = parser.parse_args()

    with open(args.trace, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        if len(header) < 4:
            sys.exit(f"Invalid trace: {args.trace}")
        M, B, num_batches, num_dense = int(header[0]), int(header[1]), int(header[2]), int(header[3])

        table_indices = [[] for _ in range(M)]

        for _ in range(num_batches):
            # Skip dense features: B * num_dense * 4 bytes
            f.seek(B * num_dense * 4, 1)

            for t in range(M):
                offsets = np.fromfile(f, dtype=np.int32, count=B + 1)
                num_indices = offsets[-1]
                idx = np.fromfile(f, dtype=np.int32, count=num_indices)
                table_indices[t].append(idx)

    print(f"Loaded trace: M={M}, B={B}, batches={num_batches}")

    # Count frequencies for each (table, row)
    items = []
    for t in range(M):
        if not table_indices[t]:
            continue
        all_idx = np.concatenate(table_indices[t])
        max_id = all_idx.max() if len(all_idx) > 0 else 0
        counts = np.bincount(all_idx)
        unique_rows = np.nonzero(counts)[0]
        row_counts = counts[unique_rows]

        # Store (count, table, row)
        t_arr = np.full_like(unique_rows, t, dtype=np.int32)
        items.append(np.column_stack((row_counts, t_arr, unique_rows)))

    if not items:
        sys.exit("No indices found in trace!")

    all_items = np.vstack(items)
    # Sort descending by count (column 0)
    order = np.argsort(-all_items[:, 0])
    sorted_items = all_items[order]

    # Extract [table, row] as int32
    hot_pairs = sorted_items[:, 1:3].astype(np.int32)

    with open(args.out, "wb") as f:
        f.write(hot_pairs.tobytes())

    print(f"Wrote {len(hot_pairs)} hot (table, row) pairs to {args.out} "
          f"({len(hot_pairs) * 8 / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
