#!/usr/bin/env python3
"""Summarize the cpu_ref result CSVs written by scripts/ghc/experiments.sh.

    python3 scripts/ghc/summarize.py /tmp/yosenl-data/results

Files are named <tag>__t<threads>.csv. Output: per tag, a thread-scaling table (throughput and speedup over the
fewest threads) and the stage breakdown at the highest thread count. Uses only the standard library.
"""

import csv
import glob
import os
import re
import sys

STAGES = ["dense_bot", "lookup", "interact_top", "total"]


def load(res_dir):
    runs = {}  # tag -> {threads: {stage: median_ms}} and metadata
    meta = {}
    for path in sorted(glob.glob(os.path.join(res_dir, "*__t*.csv"))):
        m = re.match(r"(.+)__t(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        tag, threads = m.group(1), int(m.group(2))
        with open(path) as f:
            for row in csv.DictReader(f):
                runs.setdefault(tag, {}).setdefault(threads, {})[row["stage"]] = float(row["median_ms"])
                meta[tag] = (int(row["B"]), row["L"], row["alpha"], row["D"])
    return runs, meta


def main():
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    runs, meta = load(res_dir)
    if not runs:
        print(f"no results found in {res_dir}")
        return 1

    print("cpu_ref results (median ms per batch; throughput = batch size / total)\n")
    for tag in sorted(runs):
        B, L, alpha, D = meta[tag]
        by_t = runs[tag]
        ts = sorted(by_t)
        base_t = ts[0]
        base_total = by_t[base_t].get("total")
        print(f"[{tag}]  B={B}  L(first table)={L}  alpha={alpha}  D={D}")
        print(f"  {'threads':>7} {'total ms':>10} {'samples/s':>11} {'speedup':>8}   " +
              " ".join(f"{s:>12}" for s in STAGES[:-1]))
        for t in ts:
            st = by_t[t]
            total = st.get("total")
            if total is None:
                continue
            sps = B / (total / 1000.0)
            sp = base_total / total if base_total else float("nan")
            print(f"  {t:>7} {total:>10.2f} {sps:>11.0f} {sp:>7.2f}x   " +
                  " ".join(f"{st.get(s, float('nan')):>12.2f}" for s in STAGES[:-1]))
        top = by_t[ts[-1]]
        total = top.get("total")
        if total:
            parts = ", ".join(f"{s} {100 * top[s] / total:.0f}%" for s in STAGES[:-1] if s in top)
            print(f"  stage share at {ts[-1]} threads: {parts}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
