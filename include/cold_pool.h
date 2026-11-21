#pragma once

// CPU side of gpu_tier_pool (OpenMP threads + AVX2).
//
// Splits one flattened batch into hot lookups, which the GPU pools from its cache,
// and cold lookups, which the CPU pools from the host table. For every bag k:
//   hot_offsets[k] .. hot_offsets[k+1]  cache slots of the hot lookups in bag k
//   cold_sum[k][0..D)                   sum of the cold rows in bag k (0 if none)
// remap[global_row] is the cache slot of that row, or -1 if it is not cached.
// hot_slots must have room for every lookup of the batch.
// Returns the number of hot lookups and writes the number of cold lookups to *num_cold.
int split_and_pool(const int* offsets, const int* rows, int num_bags, const int* remap, const float* host_table,
                   int D, int* hot_offsets, int* hot_slots, float* cold_sum, long long* num_cold);
