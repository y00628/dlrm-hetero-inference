#!/usr/bin/env python3
"""Build the pool vs zero-copy comparison table from the runs of scripts/ghc/phase1_eval.sh.

    python3 scripts/ghc/phase1_summary.py <OUT dir> "<fractions>"

Reads <OUT>/sweep/<config>/results.csv and the per-run logs (pool_<f>.log, zc_<f>.log). Prints markdown.
Standard library only.
"""

import csv
import os
import re
import sys


def medians(csv_path):
    """(impl, variant) -> {stage: median_ms}"""
    out = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out.setdefault((row["impl"], row.get("variant", "")), {})[row["stage"]] = float(row["median_ms"])
    return out


def grep(path, pattern):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = re.search(pattern, f.read())
    return m.groups() if m else None


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    out_dir = sys.argv[1]
    fracs = sys.argv[2].split() if len(sys.argv) > 2 else []
    sweep = os.path.join(out_dir, "sweep")
    if not os.path.isdir(sweep):
        print("no sweep results found")
        return 1

    print("# Phase 1 results: pool vs zero-copy\n")
    link = os.path.join(out_dir, "pcie_link.txt")
    if os.path.exists(link):
        print("PCIe link: " + " | ".join(l.strip() for l in open(link).read().strip().splitlines()[1:]) + "\n")

    for tag in sorted(os.listdir(sweep)):
        d = os.path.join(sweep, tag)
        res = medians(os.path.join(d, "results.csv"))
        single = res.get(("gpu_single", ""), {})
        print(f"## {tag}\n")
        if single:
            print(f"All-resident `gpu_single`: lookup {fmt(single.get('lookup'))} ms, total {fmt(single.get('total'))} ms "
                  f"per batch (embedding stage only; total includes the id copy).\n")
        print("| cache | hit rate | pool total ms | pool cpu_gather | pool h2d | pool lookup | pool H2D payload MB | "
              "cold rows as raw MB | zc total ms | zc lookup ms | zc PCIe MB | zc GB/s | faster |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|")
        for f in fracs:
            pool = res.get(("gpu_tier_pool", f"cache={f}"), {})
            zc = res.get(("gpu_tier_zerocopy", f"cache={f}"), {})
            hit = grep(os.path.join(d, f"pool_{f}.log"), r"hit rate ([0-9.]+)")
            pcie = grep(os.path.join(d, f"pool_{f}.log"), r"PCIe per batch: ([0-9.]+) MB sent \(cold rows unpooled would be ([0-9.]+) MB\)")
            if not pcie:  # New explicit payload labels; retain compatibility with old logs.
                total = grep(os.path.join(d, f"pool_{f}.log"), r"H2D payload per batch: ([0-9.]+) MB total")
                cold = grep(os.path.join(d, f"pool_{f}.log"), r"Unpooled cold-vector payload per batch: ([0-9.]+) MB")
                pcie = (total[0], cold[0]) if total and cold else None
            zrate = grep(os.path.join(d, f"zc_{f}.log"), r"zero-copy reads per batch: ([0-9.]+) MB; effective rate ([0-9.]+) GB/s")
            pt, zt = pool.get("total"), zc.get("total")
            faster = "-"
            if pt and zt:
                faster = f"pool {zt / pt:.2f}x" if pt < zt else f"zero-copy {pt / zt:.2f}x"
            print(f"| {float(f) * 100:g}% | {hit[0] if hit else '-'} | {fmt(pt)} | {fmt(pool.get('cpu_gather'))} | "
                  f"{fmt(pool.get('h2d'))} | {fmt(pool.get('lookup'))} | {pcie[0] if pcie else '-'} | "
                  f"{pcie[1] if pcie else '-'} | {fmt(zt)} | {fmt(zc.get('lookup'))} | "
                  f"{zrate[0] if zrate else '-'} | {zrate[1] if zrate else '-'} | {faster} |")
        print()
        print("Notes: times are medians per batch over 50 batches after 5 warm-up batches. `pool H2D payload MB` is what "
              "is calculated from copy sizes (hot slots, offsets and one partial sum per bag); `cold rows as raw MB` is the "
              "logical cold-vector payload, excluding metadata and cache effects, not measured bus traffic. The hot list is ranked on the first half of the trace and "
              "the runs cycle through all batches, so the hit rate is slightly optimistic (see hot_profile output for the "
              "hold-out rate).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
