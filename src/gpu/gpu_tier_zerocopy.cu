// gpu_tier_zerocopy: hot rows cached in GPU memory, full table in mapped (zero-copy) host memory.
// Per batch the CPU does no pooling at all: the GPU kernel pools every bag itself, reading hot rows from the
// GPU cache and cold rows straight from host memory across PCIe.
//
// Usage: ./bin/gpu_tier_zerocopy <config> <cache_fraction> [hot_file] [results.csv]
//   cache_fraction  fraction of all rows kept on the GPU, e.g. 0.25 (same meaning as gpu_tier_pool)
//   hot_file        int32 pairs (table, row), hottest first, written by bin/hot_profile.
//                   Without it the first rows of each table are cached (for correctness tests).
// Writes data/gpu_tier_zerocopy_<config>_b{0,1}_pooled.bin (and zc_b{0,1}_pooled.bin in the current
// directory) and appends to data/results.csv.
//
// Compare with gpu_tier_pool at the same cache_fraction and hot_file: that path pools cold rows on the CPU and
// sends one partial sum per bag, this one moves every cold row over PCIe.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// One warp per bag, lanes cover the D columns. For every lookup the row comes from the GPU cache if remap
// says it is cached, otherwise from the mapped host table. All lanes of a warp take the same branch.
// Four lookups are loaded before they are added, so several PCIe reads are in flight per warp; the adds
// stay in lookup order, so the result is the same as a plain loop.
__global__ void pool_zc_kernel(const float* __restrict__ cache, const float* host_table, const int* __restrict__ remap,
                               const int* __restrict__ offsets, const int* __restrict__ rows, int num_bags, int D,
                               float* __restrict__ out) {
    int bag = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (bag >= num_bags) return;

    int start = offsets[bag];
    int end = offsets[bag + 1];
    for (int d = lane; d < D; d += 32) {
        float sum = 0.0f;
        int i = start;
        for (; i + 4 <= end; i += 4) {
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int r = rows[i + j];
                int slot = remap[r];
                const float* src = (slot >= 0) ? cache + (i64)slot * D : host_table + (i64)r * D;
                v[j] = src[d];
            }
            sum += v[0];
            sum += v[1];
            sum += v[2];
            sum += v[3];
        }
        for (; i < end; i++) {
            int r = rows[i];
            int slot = remap[r];
            const float* src = (slot >= 0) ? cache + (i64)slot * D : host_table + (i64)r * D;
            sum += src[d];
        }
        out[(i64)bag * D + d] = sum;
    }
}

static void launch_pool_zc(const float* cache, const float* host_table, const int* remap, const int* offsets,
                           const int* rows, int num_bags, int D, float* out, cudaStream_t stream) {
    int threads = 256;  // 8 warps = 8 bags per block
    int blocks = (num_bags * 32 + threads - 1) / threads;
    pool_zc_kernel<<<blocks, threads, 0, stream>>>(cache, host_table, remap, offsets, rows, num_bags, D, out);
    CUDA_CHECK(cudaGetLastError());
}

// Global rows to cache, hottest first (same rule as gpu_tier_pool).
static vector<int> pick_cached_rows(const Config& cfg, const vector<i64>& base, i64 capacity, const char* hot_file) {
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
        printf("usage: %s <config> <cache_fraction> [hot_file] [results.csv]\n", argv[0]);
        return 1;
    }
    Config cfg = load_config(argv[1]);
    double fraction = atof(argv[2]);
    const char* hot_file = nullptr;
    string csv = "data/results.csv";
    for (int a = 3; a < argc; a++) {
        string arg = argv[a];
        if (arg.size() >= 4 && arg.substr(arg.size() - 4) == ".csv") {
            csv = arg;
        } else {
            hot_file = argv[a];
        }
    }
    int D = cfg.emb_dim;
    auto base = table_base(cfg);
    i64 total_rows = base[cfg.num_tables];

    // Must run before the CUDA context exists; harmless where unified addressing already maps host memory.
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaGetLastError();

    // CPU tier: the full table in pinned, mapped host memory. The GPU reads it through host_table_dev.
    float* host_table = nullptr;
    cudaError_t err = cudaHostAlloc(&host_table, total_rows * D * sizeof(float), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "cannot pin %.2f GB of host memory (%s); the locked-memory limit (ulimit -l) or RAM may be too small\n",
                total_rows * D * 4.0 / 1e9, cudaGetErrorString(err));
        return 1;
    }
    for (int t = 0; t < cfg.num_tables; t++) {
        fill_table(cfg.seed, t, 0, cfg.rows[t], D, host_table + base[t] * D);
    }
    float* host_table_dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&host_table_dev, host_table, 0));

    // GPU tier: cache of the chosen rows, and remap[global_row] = cache slot or -1 (also copied to the GPU)
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
    int* d_remap;
    CUDA_CHECK(cudaMalloc(&d_cache, h_cache.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cache, h_cache.data(), h_cache.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_remap, total_rows * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_remap, remap.data(), total_rows * sizeof(int), cudaMemcpyHostToDevice));
    printf("tables: %.2f GB in mapped host memory, %lld rows (%.2f GB) cached on GPU\n", total_rows * D * 4.0 / 1e9,
           num_cached, num_cached * D * 4.0 / 1e9);

    Trace tr = load_trace(cfg.trace_file);
    auto batches = flatten_trace(tr, base);
    tr.batches.clear();

    int num_bags = batches[0].num_bags;
    int max_lookups = 0;
    for (const auto& fb : batches) max_lookups = max(max_lookups, fb.num_lookups);

    // Cold lookups per batch, counted once outside the timed loop (for hit rate and PCIe traffic)
    vector<i64> cold_per_batch(batches.size(), 0);
    for (size_t b = 0; b < batches.size(); b++) {
        for (int i = 0; i < batches[b].num_lookups; i++) {
            if (remap[batches[b].rows[i]] < 0) cold_per_batch[b]++;
        }
    }

    int *d_offsets, *d_rows;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_offsets, (num_bags + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rows, max_lookups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (i64)num_bags * D * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    GpuTimer h2d_timer, lookup_timer;
    StageTimes times;
    i64 lookups_total = 0, cold_total = 0;
    double lookup_ms_total = 0;

    for (int i = 0; i < WARMUP_BATCHES + TIMED_BATCHES; i++) {
        size_t bi = i % batches.size();
        auto& fb = batches[bi];
        double t0 = now_ms();

        // PCIe: row ids only (4 bytes per lookup); the rows themselves are read by the kernel
        h2d_timer.start(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_offsets, fb.offsets, (num_bags + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_rows, fb.rows, fb.num_lookups * sizeof(int), cudaMemcpyHostToDevice, stream));
        h2d_timer.stop(stream);

        // GPU: hot rows from the cache, cold rows over PCIe
        lookup_timer.start(stream);
        launch_pool_zc(d_cache, host_table_dev, d_remap, d_offsets, d_rows, num_bags, D, d_out, stream);
        lookup_timer.stop(stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        double total = now_ms() - t0;

        if (i < 2) {
            string path = "data/gpu_tier_zerocopy_" + cfg.name + "_b" + to_string(i) + "_pooled.bin";
            dump_device(d_out, (size_t)num_bags * D, path);
            dump_device(d_out, (size_t)num_bags * D, "zc_b" + to_string(i) + "_pooled.bin");
        }
        if (i >= WARMUP_BATCHES) {
            float lookup_ms = lookup_timer.elapsed_ms();
            times["h2d"].push_back(h2d_timer.elapsed_ms());
            times["lookup"].push_back(lookup_ms);
            times["total"].push_back(total);
            lookups_total += fb.num_lookups;
            cold_total += cold_per_batch[bi];
            lookup_ms_total += lookup_ms;
        }
    }

    double hit_rate = 1.0 - (double)cold_total / lookups_total;
    double cold_mb = (double)cold_total * D * 4.0 / TIMED_BATCHES / 1e6;
    printf("hit rate %.3f, cold rows per batch %.0f\n", hit_rate, (double)cold_total / TIMED_BATCHES);
    printf("PCIe zero-copy reads per batch: %.2f MB; effective rate %.2f GB/s (cold bytes / lookup kernel time; "
           "includes the hot-row work)\n",
           cold_mb, (double)cold_total * D * 4.0 / (lookup_ms_total * 1e6));

    string variant = string("cache=") + argv[2];
    if (hot_file == nullptr) variant += " first-rows";
    append_results(csv, cfg, "gpu_tier_zerocopy", variant, 1, times);

    free_flat_batches(batches);
    cudaFreeHost(host_table);
    cudaFree(d_cache);
    cudaFree(d_remap);
    cudaFree(d_offsets);
    cudaFree(d_rows);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return 0;
}
