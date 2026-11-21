#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "config.h"

// Every implementation runs WARMUP_BATCHES untimed, then times TIMED_BATCHES.
const int WARMUP_BATCHES = 5;
const int TIMED_BATCHES = 50;

// stage name -> one time in ms per timed batch
// stage names: cpu_gather, h2d, lookup, exchange, dense_bot, interact_top, total
typedef std::map<std::string, std::vector<double>> StageTimes;

inline double now_ms() {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

// Appends one CSV row per stage (writes the header if the file is new):
//   impl,variant,config,gpus,B,alpha,L,D,mlp_top,stage,median_ms,p10_ms,p90_ms
// impl names: cpu_ref, gpu_single, gpu_tier_pool, gpu_tier_zerocopy,
//             mgpu_table_emul, mgpu_table_nccl, mgpu_row_nccl, pytorch_dlrm
// variant: free-form setting of this run without commas, e.g. "cache=0.25" ("" if none).
// In multi-GPU runs, only rank 0 calls this.
void append_results(const std::string& csv_path, const Config& cfg, const std::string& impl,
                    const std::string& variant, int gpus, const StageTimes& times);

// 5-argument backward-compatible overload for implementations without variants (defaults to variant = "")
inline void append_results(const std::string& csv_path, const Config& cfg, const std::string& impl,
                           int gpus, const StageTimes& times) {
    append_results(csv_path, cfg, impl, "", gpus, times);
}
