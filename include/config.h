#pragma once

#include <string>
#include <vector>

// One workload, read from a "key = value" file in configs/.
// tools/gen_trace.py reads the same file.
struct Config {
    std::string name;             // file name without extension, used in results CSV
    int num_tables;               // M
    std::vector<long long> rows;  // rows per table
    std::vector<int> pooling;     // pooling factor L per table
    int emb_dim;                  // D
    int batch_size;               // B
    int num_batches;
    int num_dense;                // 13 for Criteo
    double alpha;                 // Zipf skew, 0 = uniform
    unsigned long long seed;
    std::vector<int> mlp_bot;     // e.g. 13-512-256-64
    std::vector<int> mlp_top;     // e.g. 415-512-256-1
    std::string trace_file;
};

Config load_config(const std::string& path);

// Size of [bottom MLP output, pairwise dots] = D + n(n-1)/2 with n = M + 1.
int interaction_dim(const Config& cfg);
