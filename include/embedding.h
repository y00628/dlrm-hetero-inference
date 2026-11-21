#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "config.h"
#include "trace.h"

// All tables are stored back to back: global row = base[t] + row (base has M + 1 entries).
// Global rows are int, so the total row count must stay below 2^31.
std::vector<long long> table_base(const Config& cfg);

// One batch flattened into a single CSR over bags. Bag k = b * M + t, so a pooled
// output [num_bags][D] is already in [B][M][D] order. Arrays are in pinned host memory.
struct FlatBatch {
    int num_bags;
    int num_lookups;
    int* offsets;  // num_bags + 1
    int* rows;     // global row ids
};

std::vector<FlatBatch> flatten_trace(const Trace& tr, const std::vector<long long>& base);
void free_flat_batches(std::vector<FlatBatch>& batches);

// out[k][d] = init[k][d] (0 if init is null) + sum over i in bag k of table[rows[i]][d].
// One warp per bag; lanes cover the D columns so each row read is coalesced.
void launch_pool(const float* table, const int* offsets, const int* rows, int num_bags, int D,
                 const float* init, float* out, cudaStream_t stream);

// Copies count floats from the GPU and writes them as a correctness dump.
void dump_device(const float* d_data, size_t count, const std::string& path);
