# Phase 1 results: pool vs zero-copy

PCIe link: link gen seen during bench_pcie: 3 (x16); max supported: NVIDIA GeForce RTX 2080, 3, 16

## meta

All-resident `gpu_single`: lookup 0.32 ms, total 0.49 ms per batch (embedding stage only; total includes the id copy).

| cache | hit rate | pool total ms | pool cpu_gather | pool h2d | pool lookup | pool PCIe MB sent | cold rows as raw MB | zc total ms | zc lookup ms | zc PCIe MB | zc GB/s | faster |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 0% | 0.000 | 5.13 | 4.74 | 0.34 | 0.03 | 4.26 | 123.55 | 9.48 | 9.31 | 123.55 | 13.25 | pool 1.85x |
| 1% | 0.409 | 4.72 | 4.22 | 0.41 | 0.07 | 5.05 | 73.03 | 5.87 | 5.70 | 73.03 | 12.81 | pool 1.24x |
| 5% | 0.572 | 3.87 | 3.30 | 0.43 | 0.12 | 5.36 | 52.86 | 4.46 | 4.29 | 52.86 | 12.53 | pool 1.15x |
| 10% | 0.659 | 3.43 | 2.85 | 0.45 | 0.17 | 5.53 | 42.08 | 3.91 | 3.75 | 42.08 | 11.92 | pool 1.14x |
| 25% | 0.784 | 2.99 | 2.37 | 0.47 | 0.24 | 5.77 | 26.69 | 3.31 | 3.15 | 26.69 | 10.63 | pool 1.11x |
| 50% | 0.851 | 2.70 | 2.04 | 0.48 | 0.28 | 5.90 | 18.37 | 2.84 | 2.67 | 18.37 | 10.56 | pool 1.05x |
| 100% | 1.000 | 1.37 | 0.56 | 0.49 | 0.31 | 6.19 | 0.00 | 0.77 | 0.60 | 0.00 | 0.00 | zero-copy 1.79x |

Notes: times are medians per batch over 50 batches after 5 warm-up batches. `pool PCIe MB sent` is what crosses PCIe with CPU partial pooling (ids, hot slots and one partial sum per bag); `cold rows as raw MB` is what would cross without pooling, i.e. what zero-copy reads. The hot list is ranked on the first half of the trace and the runs cycle through all batches, so the hit rate is slightly optimistic (see hot_profile output for the hold-out rate).

## small

All-resident `gpu_single`: lookup 0.30 ms, total 0.49 ms per batch (embedding stage only; total includes the id copy).

| cache | hit rate | pool total ms | pool cpu_gather | pool h2d | pool lookup | pool PCIe MB sent | cold rows as raw MB | zc total ms | zc lookup ms | zc PCIe MB | zc GB/s | faster |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 0% | 0.000 | 6.50 | 6.12 | 0.34 | 0.03 | 4.26 | 134.22 | 15.27 | 15.09 | 134.22 | 8.68 | pool 2.35x |
| 1% | 0.680 | 4.99 | 4.40 | 0.45 | 0.12 | 5.69 | 42.96 | 8.90 | 8.72 | 42.96 | 5.26 | pool 1.78x |
| 5% | 0.792 | 4.26 | 3.60 | 0.47 | 0.19 | 5.92 | 27.98 | 5.74 | 5.56 | 27.98 | 5.04 | pool 1.35x |
| 10% | 0.840 | 3.99 | 3.30 | 0.48 | 0.22 | 6.02 | 21.46 | 4.59 | 4.41 | 21.46 | 5.13 | pool 1.15x |
| 25% | 0.911 | 3.58 | 2.86 | 0.50 | 0.27 | 6.17 | 11.95 | 3.99 | 3.81 | 11.95 | 4.91 | pool 1.12x |
| 50% | 0.959 | 3.11 | 2.36 | 0.50 | 0.29 | 6.27 | 5.45 | 2.47 | 2.29 | 5.45 | 3.78 | zero-copy 1.26x |
| 100% | 1.000 | 2.35 | 1.54 | 0.51 | 0.29 | 6.36 | 0.00 | 0.74 | 0.56 | 0.00 | 0.00 | zero-copy 3.17x |

Notes: times are medians per batch over 50 batches after 5 warm-up batches. `pool PCIe MB sent` is what crosses PCIe with CPU partial pooling (ids, hot slots and one partial sum per bag); `cold rows as raw MB` is what would cross without pooling, i.e. what zero-copy reads. The hot list is ranked on the first half of the trace and the runs cycle through all batches, so the hit rate is slightly optimistic (see hot_profile output for the hold-out rate).

