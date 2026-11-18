#pragma once

// Model parameters are computed from the config seed instead of being stored.
// Every implementation (CPU reference, GPU, each multi-GPU rank) calls these
// functions on the host and gets the same values, then copies them to the GPU if needed.

#include <cmath>
#include <cstdint>

// splitmix64 finalizer
inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

// Pseudo-random float in [-0.5, 0.5) determined by (seed, a, b, c).
inline float hash_float(uint64_t seed, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t h = mix64(seed ^ a);
    h = mix64(h ^ b);
    h = mix64(h ^ c);
    return (h >> 40) * (1.0f / 16777216.0f) - 0.5f;
}

// Fills rows [row_begin, row_begin + num_rows) of one table, row-major [num_rows][D].
inline void fill_table(uint64_t seed, int table, long long row_begin, long long num_rows, int D, float* out) {
#pragma omp parallel for
    for (long long r = 0; r < num_rows; r++) {
        for (int c = 0; c < D; c++) {
            out[r * D + c] = hash_float(seed, table, row_begin + r, c);
        }
    }
}

// Weights of one MLP layer, row-major [out_dim][in_dim], scaled by 1/sqrt(in_dim).
// Layer ids: bottom layer i -> i, top layer i -> 100 + i. The model has no biases.
inline void fill_mlp_layer(uint64_t seed, int layer, int out_dim, int in_dim, float* W) {
    float scale = 1.0f / std::sqrt((float)in_dim);
    for (int o = 0; o < out_dim; o++) {
        for (int i = 0; i < in_dim; i++) {
            W[(long long)o * in_dim + i] = hash_float(seed, 1000 + layer, o, i) * scale;
        }
    }
}
