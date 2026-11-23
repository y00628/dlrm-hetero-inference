# What is done (Yosen), as of Nov 22 night

Branch `yosen/phase1`. `[verified]` means run and checked; `[written]` means the code exists but has not been run where it matters.

## Phase 0

- **CPU reference** `src/ref/cpu_ref.cpp` [verified]. Multithreaded OpenMP + AVX2. Matches an independent NumPy model on `tiny`. On `ghc39`, `scripts/ghc/verify.sh` passes 12/12: 0 warnings, 1 thread == 8 threads (1e-4), repeat runs bit-identical, checker rejects NaN and truncated files, `trace_info` OK on tiny, small, meta and criteo.
- **Data converters** [verified on real data]
  - `tools/convert_criteo.py`: text or Hugging Face `.parquet`. Criteo 1TB click-logs slice: 26 tables, L=1.
  - `tools/convert_meta.py`: real `fbgemm_t856_bs65536_0.pt`. Table-major layout confirmed, float32 indices exact, empty table skipped. 8 tables, 63 MB trace.
  - `tools/convert_mlperf.py` [written]: tested on synthetic files only. Skipped for now (the real set is ~163 GB).
  - `tools/trace_writer.py`: shared writer.
- **Machine facts** (`ghc39`): i7-9700, 8 cores, no hyperthreading, L3 12 MB, 15 GB RAM, one NUMA node; RTX 2080 8 GB (`sm_75`), driver 550.67, `nvcc` 11.7. PSC access is granted, with no nodes reserved. It is not needed until Phase 2.

## CPU baseline [verified, 8 threads, B=2048]

| config | 1 thread | 8 threads | speedup | samples/s at 8 |
|---|---|---|---|---|
| small | 103.2 ms | 17.8 ms | 5.80x | 115k |
| meta | 97.3 ms | 17.2 ms | 5.66x | 119k |
| criteo | 116.0 ms | 19.9 ms | 5.83x | 103k |

- Dense stages are about 70% of CPU time on `small` (dense_bot 33%, interact_top 36%, lookup 31%).
- Lookup time vs pooling L = 1/4/16/64: 0.49 / 1.11 / 3.14 / 10.09 ms.
- Skew: alpha 0 and 0.5 lookup 7.9 ms, alpha 1.0 5.6 ms.
- The sweeps are single runs without error bars.

## Phase 1

- **Hot-row profiler** `src/tools/hot_profile.cpp` -> `bin/hot_profile` [verified on Mac]. Ranks rows by lookup count, writes the hot file that `gpu_tier_pool` reads, and prints hit rate per cache size (in-sample and hold-out). Hot file and hit rates match an independent NumPy computation exactly. On a Zipf `small` trace: 1% cache -> 68% hit; 50% cache -> 92% hold-out.
- **`gpu_tier_zerocopy`** `src/gpu/gpu_tier_zerocopy.cu` [compiled and checked on `tiny` on GHC]. Hot rows are cached on the GPU and the whole table sits in mapped host memory. One kernel pools every bag, and the CPU does no per-batch work. On `tiny` the output is bit-identical to the CPU reference (max error 0). Not yet checked at `small` scale.
- **PCIe microbenchmark** (`bench_pcie`, run by Louis on `ghc34`): VRAM 394 GB/s, pageable H2D 10.4, pinned H2D 12.9, zero-copy streaming read 12.86 GB/s. r = 30.5x. A streaming rate is an upper bound for random row gathers.
- **Scripts** (`scripts/ghc/`)
  - `env.sh`, `prepare_data.sh`: rebuild the venv, folders and real-data traces after a `/tmp` wipe [verified].
  - `verify.sh`, `experiments.sh`, `summarize.py`: CPU checks and baseline [verified].
  - `phase1_eval.sh`, `phase1_summary.py` [written, not run]: correctness of `gpu_single`, `gpu_tier_pool` and `gpu_tier_zerocopy` against `cpu_ref`, then the pool vs zero-copy sweep and a summary table.

## Results from `phase1_eval.sh` (ghc39, Nov 23; two full runs agree)

- **Correctness: 51 passed, 0 failed.** `gpu_single`, `gpu_tier_pool` and `gpu_tier_zerocopy` match `cpu_ref` on tiny, small and meta at cache 0, 0.25 and 1.0, and pool == zero-copy.
- **PCIe link:** Gen3 x16 under load (idles at Gen1). bench: VRAM 394, pageable 10.2, pinned 12.8, zero-copy stream 12.8 GB/s.
- **Pool vs zero-copy (median ms per batch, `small`, B=2048, 8 threads):**

| cache | hit rate | pool | zero-copy | PCIe MB pool / raw cold |
|---:|---:|---:|---:|---:|
| 0% | 0.00 | 6.5 | 15.3 | 4.3 / 134 |
| 1% | 0.68 | 5.0 | 8.9 | 5.7 / 43 |
| 10% | 0.84 | 4.0 | 4.6-5.4 | 6.0 / 21 |
| 50% | 0.96 | 3.1 | 2.2-2.5 | 6.3 / 5.5 |
| 100% | 1.00 | 2.4 | 0.74 | 6.4 / 0 |

  `meta` is the same shape; pool wins until 100% cache.
- **Reading:** CPU partial pooling cuts PCIe traffic 31x at an empty cache (4.3 vs 134 MB), and the CPU gathers at about 22 GB/s, faster than PCIe. Zero-copy only wins once the hit rate is above about 95%. Random 256 B zero-copy reads reach 4-9 GB/s on `small`, below the 12.8 GB/s streaming ceiling. Zero-copy timings vary about +-15% between runs; pool timings vary under 3%.
- Against the CPU-only lookup (5.5 ms on `small`), tiering helps once the cache reaches about 25%: 3.6 ms at 25%, 3.1 ms at 50%, 0.49 ms all-resident.
- Real Meta data has far lower hit rates than synthetic Zipf: 1% cache gives 0.40 on meta vs 0.68 on small.
- The zero-copy GB/s on meta (up to 13.3) exceeds the PCIe stream limit, so it is inflated by L2 hits on repeated rows. Do not quote it as PCIe throughput.

## Not verified yet

- A dedicated random-row gather microbenchmark (Louis's item).
- Zero-copy variance: only two runs per point.

## Not done

- Dense stage (cuBLAS MLPs and dot interaction), so no end-to-end CTR.
- Milestone report page in `docs/` (due Nov 23, 11:59 PM, tonight).
- Random-row gather microbenchmark (Louis's item).
- NVIDIA DLRM baseline.
- Real-file MLPerf conversion.
- Louis's remaining items: the `include/` interfaces for the dense and exchange stages, the CSV `bytes` column, and the NCCL Makefile target.

## Git state

- Uncommitted: `Makefile` (two new targets), `src/tools/hot_profile.cpp`, `src/gpu/gpu_tier_zerocopy.cu`, `scripts/ghc/phase1_eval.sh`, `scripts/ghc/phase1_summary.py`.
- Local only, not for pushing: `CHECKLIST.md`, `STATUS.md`.
- Stash `stash@{0}` (on the old phase-0 branch): the `results/cpu_baseline` copy in `experiments.sh` and the `WORK` fix in `prepare_data.sh`. Not re-applied yet.
