// gpu_single: all tables resident in GPU memory, embedding stage only.
// Usage: ./bin/gpu_single configs/small.cfg
// Writes data/gpu_single_<config>_b{0,1}_pooled.bin and appends to data/results.csv.

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "config.h"
#include "cuda_utils.h"
#include "embedding.h"
#include "init.h"
#include "timing.h"
#include "trace.h"

using namespace std;
using i64 = long long;

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: %s <config>\n", argv[0]);
        return 1;
    }
    Config cfg = load_config(argv[1]);
    int D = cfg.emb_dim;
    auto base = table_base(cfg);
    i64 total_rows = base[cfg.num_tables];

    // Tables: fill one table at a time on the host and copy it into one device array
    float* d_table;
    CUDA_CHECK(cudaMalloc(&d_table, total_rows * D * sizeof(float)));
    for (int t = 0; t < cfg.num_tables; t++) {
        vector<float> h(cfg.rows[t] * D);
        fill_table(cfg.seed, t, 0, cfg.rows[t], D, h.data());
        CUDA_CHECK(cudaMemcpy(d_table + base[t] * D, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    printf("tables: %.2f GB on GPU\n", total_rows * D * 4.0 / 1e9);

    Trace tr = load_trace(cfg.trace_file);
    auto batches = flatten_trace(tr, base);
    tr.batches.clear();

    int num_bags = batches[0].num_bags;
    int max_lookups = 0;
    for (const auto& fb : batches) max_lookups = max(max_lookups, fb.num_lookups);

    int *d_offsets, *d_rows;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_offsets, (num_bags + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rows, max_lookups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (i64)num_bags * D * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    GpuTimer h2d_timer, lookup_timer;
    StageTimes times;
    double gather_bytes = 0, gather_ms = 0;

    for (int i = 0; i < WARMUP_BATCHES + TIMED_BATCHES; i++) {
        auto& fb = batches[i % batches.size()];
        double t0 = now_ms();

        h2d_timer.start(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_offsets, fb.offsets, (num_bags + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_rows, fb.rows, fb.num_lookups * sizeof(int), cudaMemcpyHostToDevice, stream));
        h2d_timer.stop(stream);

        lookup_timer.start(stream);
        launch_pool(d_table, d_offsets, d_rows, num_bags, D, nullptr, d_out, stream);
        lookup_timer.stop(stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        double total = now_ms() - t0;

        if (i < 2) {
            string path = "data/gpu_single_" + cfg.name + "_b" + to_string(i) + "_pooled.bin";
            dump_device(d_out, (size_t)num_bags * D, path);
        }
        if (i >= WARMUP_BATCHES) {
            float lookup_ms = lookup_timer.elapsed_ms();
            times["h2d"].push_back(h2d_timer.elapsed_ms());
            times["lookup"].push_back(lookup_ms);
            times["total"].push_back(total);
            gather_bytes += (double)fb.num_lookups * D * sizeof(float);
            gather_ms += lookup_ms;
        }
    }

    printf("gather bandwidth: %.1f GB/s (embedding rows read / lookup kernel time)\n", gather_bytes / gather_ms / 1e6);
    append_results("data/results.csv", cfg, "gpu_single", "", 1, times);

    free_flat_batches(batches);
    cudaFree(d_table);
    cudaFree(d_offsets);
    cudaFree(d_rows);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return 0;
}
