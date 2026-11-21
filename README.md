# dlrm-hetero-inference

15-418/618 Final Project: Scalable DLRM Inference across GPUs and CPU Memory.
Project web page sources are in `docs/`.

## Layout

```
include/        config, trace, init (model parameters), timing, check, cuda_utils
src/common/     shared code
src/gpu/        embedding kernel, gpu_single, gpu_tier_pool
src/cpu/        CPU cold-row pooling for the tier (OpenMP + AVX2)
src/tools/      trace_info (check a trace), check (compare output dumps)
tools/          gen_trace.py (synthetic Zipf traces, needs numpy)
configs/        tiny (correctness), small (fits one RTX 2080), medium (10 GB, needs CPU tier)
data/           generated traces (not committed)
```

## Build and run

```
make                                          # GHC; use "make ARCH=sm_70" on PSC
python3 tools/gen_trace.py configs/tiny.cfg
./bin/trace_info configs/tiny.cfg             # prints OK if the trace matches the config
./bin/gpu_single configs/tiny.cfg            # all tables on the GPU
./bin/gpu_tier_pool configs/tiny.cfg 0.3      # 30% of rows cached on the GPU, rest pooled on the CPU
./bin/check data/gpu_single_tiny_b0_pooled.bin data/gpu_tier_pool_tiny_b0_pooled.bin
```

Results are appended to `data/results.csv`.

## Model definition (all implementations must match)

- Bottom MLP: `x = dense[b]`; for every layer `x = relu(W x)`. Output `bot[b]` has D values.
- Embeddings: `e_m[b]` = sum of the rows of table m listed in bag b.
- Interaction: `z[b] = [bot[b], dot(v_i, v_j) for i < j]` with `v = (bot[b], e_0[b], ..., e_{M-1}[b])`,
  pairs in order (0,1), (0,2), ..., (1,2), ... Size `D + n(n-1)/2`, `n = M + 1`.
- Top MLP: `z = relu(W z)` for every layer except the last, which has no relu.
- Output: `ctr[b] = sigmoid(top output)`.
- No biases. Weights and embedding rows come from `include/init.h`.

## Conventions

- Layouts are row-major: dense `[B][num_dense]`, pooled `[B][M][D]`, ctr `[B]`.
- Correctness: dump pooled and ctr for batches 0 and 1 (`include/check.h`);
  pass if `|x - ref| <= 1e-4 * max(|ref|, 1)`.
- Timing: 5 warm-up batches, 50 timed batches, results appended with `append_results` (`include/timing.h`).
