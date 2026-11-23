# Project checklist

Last updated: Nov 22. `origin/main` is at 0a4967e (PR #4 merged: converters, edited meta config, scripts). Owners follow PLAN.md.
`[x]` done, `[~]` partly done or untested, `[ ]` not started, `[?]` status unknown, confirm with Louis.

## Deadlines
- Milestone report: **Nov 23, 11:59 PM**
- Final report: Dec 10, 11:59 PM. Poster: Dec 11.

## Phase 0: setup and interfaces
Louis
- [x] Config parsing, hash-based init (`init.h`), timer, results CSV writer, checker
- [x] Zipf trace generator (`tools/gen_trace.py`)
- [~] Makefile: `ARCH` flag done; no `mgpu` NCCL target yet
- [ ] Interfaces in `include/`: embedding stage, dense stage, exchange layer
- [ ] CSV `bytes` column (PLAN.md wants it; `timing.h` does not write it)

Yosen
- [x] Serial/multithreaded CPU reference `src/ref/cpu_ref.cpp` (matched NumPy model on `tiny`, scalar build on macOS)
- [x] CPU reference with real flags (OpenMP + AVX2) on GHC: `scripts/ghc/verify.sh` passed 12/12 on `ghc39` (Nov 22): 0 warnings, 1 vs 8 threads agree, repeat run bit-identical, `trace_info` OK on tiny/small/meta/criteo. Optional: compare against Mac scalar dumps (`MAC_DUMPS`).
- [x] Data converters (merged, PR #4) `tools/convert_criteo.py` (text or `.parquet`), `tools/convert_meta.py`, `tools/convert_mlperf.py`, shared `tools/trace_writer.py`
- [x] Criteo: Kaggle files are gone and the official link is dead (host no longer resolves). Substitute: Hugging Face `criteo/CriteoClickLogs` (1TB click logs, same layout, CC BY-NC-SA 4.0). Converted one real 48 MB part file on GHC (Nov 22): 26 tables, L=1, 32 MB trace, `trace_info` OK. Say "Criteo 1TB click logs (slice)" in the report, not "Kaggle".
- [x] Meta converter run on real `fbgemm_t856_bs65536_0.pt` (Nov 22): table-major layout confirmed, float32 indices exact (max 12.5M < 2^24), empty table 1 skipped. 8 tables, 63 MB trace; `trace_info` OK on GHC. Converter now also stops on non-integer indices.
- [~] `convert_mlperf.py` written, tested on synthetic npz only (format notes below). Skipped for the milestone: the preprocessed set is ~163 GB and the downloader has no file selection.
- [ ] Run `convert_mlperf.py` on the real `_debug` or `_sample` files; confirm dtypes, dense range, and that npz members are uncompressed
- [ ] NVIDIA DLRM baseline: environment and first run (fallback: `facebookresearch/dlrm` after ~2 days)

Both
- [x] PSC GPU access granted; no nodes reserved. Not needed until Phase 2 (multi-GPU), so skip for the milestone.
- [x] Machine facts, GHC `ghc39` (Nov 22): i7-9700 8 cores (no HT), L2 256 KB/core, L3 12 MB, 15 GB RAM, 1 NUMA node; RTX 2080 8 GB (`sm_75`), driver 550.67, `nvcc` 11.7; `ulimit -l` ~1.9 GiB (check it does not cap pinned memory). Still missing: PCIe gen/width (`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv`). PSC facts later, with Phase 2.

Gate: both machines build; one Kaggle and one Meta trace load; serial reference produces dumps. Status: met on GHC (Nov 22). Only Louis's interfaces in `include/` are open.

## Phase 1: single GPU + CPU tier (milestone content)
Louis
- [x] GPU embedding lookup + pooling kernel (`gpu_single`): matches reference on tiny/small/meta (Nov 23, GHC); all-resident lookup 0.30 ms (small), 0.32 ms (meta)
- [x] Tiered path `gpu_tier_pool` matches reference at cache 0 / 0.25 / 1.0 on tiny/small/meta (Nov 23, GHC)
- [~] PCIe microbenchmark `bin/bench_pcie` run by Louis on ghc34 (RTX 2080, 256 MB): VRAM 394 GB/s, pageable H2D 10.4, pinned H2D 12.9, zero-copy kernel read 12.86 GB/s (sequential stream, an upper bound for random row gathers); r = 30.5x. Still missing: a random-row gather benchmark, and a run on ghc39
- [~] Cold-row throughput: the `phase1_eval.sh` sweep gives effective zero-copy rates (small 4-9 GB/s on random rows vs 12.8 GB/s streaming); a dedicated random-gather microbenchmark is still missing

Yosen
- [ ] Dense stage: cuBLAS bottom/top MLP and dot interaction behind the dense-stage interface
- [x] Tiered path `gpu_tier_zerocopy` (matches reference, 51/51 checks with the other paths) and hot-row profiler `bin/hot_profile`
- [x] Hot-list hit rates per cache size (hold-out): small 1%/10%/50% = 0.68/0.82/0.92; meta 1%/10%/50% = 0.40/0.59/0.73
- [x] CPU baseline numbers (GHC, Nov 22, 8 threads, B=2048): `small` 103.2 ms (1 thread) -> 17.8 ms (8), 5.80x, 115k samples/s; `meta` 5.66x, `criteo` 5.83x. Dense stages are ~70% of CPU time on `small`; lookup 31%. Pooling L=1/4/16/64: lookup 0.49/1.11/3.14/10.09 ms. Skew: alpha 0 and 0.5 lookup 7.9 ms, alpha 1.0 5.6 ms. Sweeps are single runs. Full table: `results/cpu_baseline/summary.txt` (copy it into the repo).

Both
- [x] `gpu_single` and both tiered paths match the reference on the embedding stage (pooled dumps): 51 passed, 0 failed on GHC, Nov 23 (`scripts/ghc/phase1_eval.sh`)
- [ ] End-to-end CTR matches once the dense stage is in
- [ ] Milestone report page in `docs/` (the site links it but it does not exist): progress, first numbers, what is late, revised schedule

Gate: embedding stage matches reference (met); first pool vs zero-copy numbers (met, `results/gpu/`: pool wins at low and mid cache sizes, zero-copy only at high hit rates).

## Later phases (not started)
- Phase 2 (Nov 23-29): table-wise sharding with NCCL (Louis); all-to-all benchmark, multi-GPU correctness checks, Slurm scripts, NVIDIA baseline on PSC (Yosen)
- Phase 3 (Nov 30-Dec 6): two-stream overlap, NUMA binding (Louis); overlap analysis, row-wise sharding, sweeps and plots (Yosen)
- Phase 4 (Dec 7-11): reruns, plots, report, poster

## Scripts (`scripts/ghc/`, merged to main in PR #4)
- [x] `verify.sh`: build, `trace_info`, 1 vs N threads, repeat run, checker negative tests, optional `MAC_DUMPS` cross-build compare; prints PASS/FAIL, exit code 1 on any failure. Passed 12/12 locally.
- [x] `experiments.sh` + `summarize.py`: thread scaling, pooling sweep (L = 1,4,16,64), skew sweep (alpha = 0,0.5,1.0); one CSV per run in `$WORK/results`, summary table at the end. Tested locally on tiny/meta.
- [x] `verify.sh` run on GHC, 12/12. [x] `experiments.sh` run on GHC (Nov 22). Keep `summary.txt` and `machine.txt` in `results/cpu_baseline/` for the report. Mac timings are scalar and single-thread; do not use them.
- `env.sh` (shared setup: venv, work dirs, dangling `data/` symlink) and `prepare_data.sh` (downloads Criteo, converts Criteo and Meta, installs torch) rebuild everything after a `/tmp` wipe. The only manual step is copying the Meta `.gz` from the Mac with `scp`.
- Known limit: the results CSV records only the first table's pooling factor in the `L` column.

## Notes
- Tiny CTR values only span 0.44-0.55 (small random weights), so the CTR check is weak. Always compare the `pooled` dumps too.
- `config.cpp` aliasing bug (`assign(n, v[0])`) is fixed on `main`.
- Trace layout, output dump names and the 1e-4 tolerance are fixed in README.md; do not change them without telling the other person.

## MLPerf DLRM v2 multi-hot Criteo: format notes
Researched Nov 21 from the mlcommons inference and training repos. Not checked against real files. The training repo path is
`retired_benchmarks/recommendation_v2/torchrec_dlrm` (the old `recommendation_v2/...` path is a 404).
- **Files (inference benchmark uses day 23 only):** `day_23_dense.npy`, `day_23_labels.npy`, `day_23_sparse_multi_hot.npz`.
  Smaller variants: `*_sample` (download location unclear) and `*_debug` (~1 GB, from `tools/make_fake_criteo.sh`).
- **Sparse layout is NOT CSR.** One `.npz` with keys `"0"`..`"25"` (26 tables); each is a dense int32 array `[N][multi_hot_size_j]`.
  Bag width is fixed per table, so offsets are `arange(B+1) * width`. Members must be uncompressed to memmap (the loader asserts int32).
- **Multi-hot sizes per table:** `3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1` (214 IDs per sample in total).
- **Rows per table (hash sizes):** `40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36`.
  Four tables hit the 40M cap. At D=128 one such table is ~20 GB, so the converter must compact IDs and keep only a few tables or a slice.
- **Embedding dim is 128** in the MLPerf model; our configs use 64. Decide which to use, since it doubles table and PCIe size.
- **Where the extra IDs come from:** the multi-hot set is synthetic. Per table, a lookup table maps each original ID `x` to `[x, r_1(x), ...]`, where
  the `r_k(x)` are uniform random in `[0, rows)` (`np.random.seed(0)`, `--multi_hot_distribution_type uniform`). Skew comes only from how often each
  original ID occurs; the extra IDs are scattered across the table. Mention this in the report when calling it "realistic access patterns".
- **Dense/labels:** dense is included; dtype, shape and any transform are not confirmed (13 features expected). Check the real file before converting.
- **Access:** raw Criteo 1TB is behind a CC BY-NC-SA terms acceptance at ailab.criteo.com (a person has to accept it). Preprocessed inference set via
  `mlc-r2-downloader` from `https://inference.mlcommons-storage.org/metadata/dlrm-v2-preprocessed-dataset.uri`; total size not found.
  Sources disagree on size (343 GB, ~100 GB, 3.8-4 TB), so check the file list before downloading.

## To ask Louis tomorrow
- Who owns the GPU lookup kernel and has any phase-1 code been written (nothing is on the remote)?
- Interfaces in `include/`: who writes them, and are the dense-stage signatures agreed?
- PSC access status.
- Who writes which part of the milestone report?
