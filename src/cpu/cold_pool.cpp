#include "cold_pool.h"

#include <cstring>

#include <immintrin.h>

using i64 = long long;

// sum[0..D) += row[0..D), 8 floats per AVX2 instruction
static inline void add_row(float* sum, const float* row, int D) {
    int d = 0;
    for (; d + 8 <= D; d += 8) {
        __m256 s = _mm256_loadu_ps(sum + d);
        __m256 r = _mm256_loadu_ps(row + d);
        _mm256_storeu_ps(sum + d, _mm256_add_ps(s, r));
    }
    for (; d < D; d++) sum[d] += row[d];
}

int split_and_pool(const int* offsets, const int* rows, int num_bags, const int* remap, const float* host_table,
                   int D, int* hot_offsets, int* hot_slots, float* cold_sum, long long* num_cold) {
    // Pass 1: count hot lookups per bag
#pragma omp parallel for schedule(static)
    for (int k = 0; k < num_bags; k++) {
        int count = 0;
        for (int i = offsets[k]; i < offsets[k + 1]; i++) {
            if (remap[rows[i]] >= 0) count++;
        }
        hot_offsets[k + 1] = count;
    }

    // Prefix sum gives where each bag's hot slots start
    hot_offsets[0] = 0;
    for (int k = 0; k < num_bags; k++) hot_offsets[k + 1] += hot_offsets[k];

    // Pass 2: write hot slots and pool cold rows
    long long cold = 0;
#pragma omp parallel for schedule(static) reduction(+ : cold)
    for (int k = 0; k < num_bags; k++) {
        float* sum = cold_sum + (i64)k * D;
        memset(sum, 0, D * sizeof(float));
        int h = hot_offsets[k];
        for (int i = offsets[k]; i < offsets[k + 1]; i++) {
            int slot = remap[rows[i]];
            if (slot >= 0) {
                hot_slots[h] = slot;
                h++;
            } else {
                add_row(sum, host_table + (i64)rows[i] * D, D);
                cold++;
            }
        }
    }

    *num_cold = cold;
    return hot_offsets[num_bags];
}
