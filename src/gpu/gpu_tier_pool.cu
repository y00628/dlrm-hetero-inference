// gpu_tier_pool: hot rows cached in GPU memory, full table in host memory.
// Per batch: CPU threads split lookups into hot/cold and sum the cold rows of each bag (AVX2),
// one partial sum per bag crosses PCIe, and the GPU adds the hot rows on top.
//
// Usage: ./bin/gpu_tier_pool <config> <cache_fraction> [hot_file]
//   cache_fraction  fraction of all rows kept on the GPU, e.g. 0.25
//   hot_file        int32 pairs (table, row), hottest first (from the hot-row profiler).
//                   Without it the first rows of each table are cached (for correctness tests).
// Writes data/gpu_tier_pool_<config>_b{0,1}_pooled.bin and appends to data/results.csv.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cold_pool.h"
#include "config.h"
#include "cuda_utils.h"
#include "embedding.h"
#include "init.h"
#include "timing.h"
#include "trace.h"

using namespace std;
using i64 = long long;

// Global rows to cache, hottest first.
static vector<int> pick_cached_rows(const Config& cfg, const vector<i64>& base, i64 capacity,
                                    const char* hot_file) {
    vector<int> rows;
    if (hot_file != nullptr) {
        FILE* f = fopen(hot_file, "rb");
        if (!f) {
            fprintf(stderr, "cannot open %s\n", hot_file);
            exit(1);
        }
        int pair[2];  // table, row
        while ((i64)rows.size() < capacity && fread(pair, sizeof(int), 2, f) == 2) {
            rows.push_back((int)(base[pair[0]] + pair[1]));
        }
        fclose(f);
    } else {
        double fraction = (double)capacity / base[cfg.num_tables];
        for (int t = 0; t < cfg.num_tables; t++) {
            i64 n = (i64)(fraction * cfg.rows[t]);
            for (i64 r = 0; r < n; r++) rows.push_back((int)(base[t] + r));
        }
    }
    return rows;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <config> <cache_fraction> [hot_file]\n", argv[0]);
        return 1;
    }
    Config cfg = load_config(argv[1]);
    double fraction = atof(argv[2]);
    const char* hot_file = (argc > 3) ? argv[3] : nullptr;
    int D = cfg.emb_dim;
    auto base = table_base(cfg);
    i64 total_rows = base[cfg.num_tables];

    // CPU tier: the full table in host memory
    float* host_table = (float*)malloc(total_rows * D * sizeof(float));
    if (!host_table) {
        fprintf(stderr, "cannot allocate %.2f GB of host memory\n", total_rows * D * 4.0 / 1e9);
        return 1;
    }
    for (int t = 0; t < cfg.num_tables; t++) {
        fill_table(cfg.seed, t, 0, cfg.rows[t], D, host_table + base[t] * D);
    }

    // GPU tier: cache of the chosen rows, and remap[global_row] = cache slot or -1
    i64 capacity = (i64)(fraction * total_rows);
    auto cached = pick_cached_rows(cfg, base, capacity, hot_file);
    i64 num_cached = cached.size();
    vector<int> remap(total_rows, -1);
    vector<float> h_cache(max(num_cached, 1LL) * D);
    for (i64 s = 0; s < num_cached; s++) {
        remap[cached[s]] = (int)s;
        memcpy(&h_cache[s * D], host_table + (i64)cached[s] * D, D * sizeof(float));
    }
    float* d_cache;
    CUDA_CHECK(cudaMalloc(&d_cache, h_cache.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cache, h_cache.data(), h_cache.size() * sizeof(float), cudaMemcpyHostToDevice));
    printf("tables: %.2f GB in host memory, %lld rows (%.2f GB) cached on GPU\n", total_rows * D * 4.0 / 1e9,
           num_cached, num_cached * D * 4.0 / 1e9);

    Trace tr = load_trace(cfg.trace_file);
    auto batches = flatten_trace(tr, base);
    tr.batches.clear();

    int num_bags = batches[0].num_bags;
    int max_lookups = 0;
    for (const auto& fb : batches) max_lookups = max(max_lookups, fb.num_lookups);

    // Pinned staging buffers written by the CPU, and their device copies
    int *h_hot_offsets, *h_hot_slots;
    float* h_cold_sum;
    CUDA_CHECK(cudaMallocHost(&h_hot_offsets, (num_bags + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_hot_slots, max_lookups * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_cold_sum, (i64)num_bags * D * sizeof(float)));

    int *d_hot_offsets, *d_hot_slots;
    float *d_cold_sum, *d_out;
    CUDA_CHECK(cudaMalloc(&d_hot_offsets, (num_bags + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hot_slots, max_lookups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cold_sum, (i64)num_bags * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (i64)num_bags * D * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    GpuTimer h2d_timer, lookup_timer;
    StageTimes times;
    long long hot_total = 0, cold_total = 0;
    double pcie_bytes = 0;

    for (int i = 0; i < WARMUP_BATCHES + TIMED_BATCHES; i++) {
        auto& fb = batches[i % batches.size()];
        double t0 = now_ms();

        // CPU: split hot/cold and pool cold rows into h_cold_sum
        long long num_cold = 0;
        int num_hot = split_and_pool(fb.offsets, fb.rows, num_bags, remap.data(), host_table, D, h_hot_offsets,
                                     h_hot_slots, h_cold_sum, &num_cold);
        double cpu_ms = now_ms() - t0;

        // PCIe: hot slots and one partial sum per bag
        h2d_timer.start(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_hot_offsets, h_hot_offsets, (num_bags + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_hot_slots, h_hot_slots, num_hot * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cold_sum, h_cold_sum, (i64)num_bags * D * sizeof(float), cudaMemcpyHostToDevice, stream));
        h2d_timer.stop(stream);

        // GPU: out = cold partial sum + hot rows from the cache
        lookup_timer.start(stream);
        launch_pool(d_cache, d_hot_offsets, d_hot_slots, num_bags, D, d_cold_sum, d_out, stream);
        lookup_timer.stop(stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        double total = now_ms() - t0;

        if (i < 2) {
            string path = "data/gpu_tier_pool_" + cfg.name + "_b" + to_string(i) + "_pooled.bin";
            dump_device(d_out, (size_t)num_bags * D, path);
        }
        if (i >= WARMUP_BATCHES) {
            times["cpu_gather"].push_back(cpu_ms);
            times["h2d"].push_back(h2d_timer.elapsed_ms());
            times["lookup"].push_back(lookup_timer.elapsed_ms());
            times["total"].push_back(total);
            hot_total += num_hot;
            cold_total += num_cold;
            pcie_bytes += (num_bags + 1 + num_hot) * 4.0 + (double)num_bags * D * 4.0;
        }
    }

    double hit_rate = (double)hot_total / (hot_total + cold_total);
    double raw_cold_mb = (double)cold_total * D * 4.0 / TIMED_BATCHES / 1e6;
    printf("hit rate %.3f, cold rows per batch %.0f\n", hit_rate, (double)cold_total / TIMED_BATCHES);
    printf("PCIe per batch: %.2f MB sent (cold rows unpooled would be %.2f MB)\n", pcie_bytes / TIMED_BATCHES / 1e6,
           raw_cold_mb);

    string variant = string("cache=") + argv[2];
    if (hot_file == nullptr) variant += " first-rows";
    append_results("data/results.csv", cfg, "gpu_tier_pool", variant, 1, times);

    free_flat_batches(batches);
    free(host_table);
    cudaFreeHost(h_hot_offsets);
    cudaFreeHost(h_hot_slots);
    cudaFreeHost(h_cold_sum);
    cudaFree(d_cache);
    cudaFree(d_hot_offsets);
    cudaFree(d_hot_slots);
    cudaFree(d_cold_sum);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return 0;
}
