#include "timing.h"

#include <algorithm>
#include <cstdio>

// p-th percentile (0..1) of an already sorted list
static double percentile(const std::vector<double>& sorted, double p) {
    return sorted[(size_t)(p * (sorted.size() - 1) + 0.5)];
}

void append_results(const std::string& csv_path, const Config& cfg, const std::string& impl,
                    const std::string& variant, int gpus, const StageTimes& times) {
    FILE* check = fopen(csv_path.c_str(), "r");
    bool new_file = (check == nullptr);
    if (check) fclose(check);

    FILE* f = fopen(csv_path.c_str(), "a");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", csv_path.c_str());
        return;
    }
    if (new_file) fprintf(f, "impl,variant,config,gpus,B,alpha,L,D,mlp_top,stage,median_ms,p10_ms,p90_ms\n");

    std::string mlp_top;
    for (size_t i = 0; i < cfg.mlp_top.size(); i++) {
        if (i > 0) mlp_top += "-";
        mlp_top += std::to_string(cfg.mlp_top[i]);
    }

    for (auto it = times.begin(); it != times.end(); ++it) {
        std::vector<double> v = it->second;
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        fprintf(f, "%s,%s,%s,%d,%d,%g,%d,%d,%s,%s,%.4f,%.4f,%.4f\n", impl.c_str(), variant.c_str(), cfg.name.c_str(), gpus,
                cfg.batch_size, cfg.alpha, cfg.pooling[0], cfg.emb_dim, mlp_top.c_str(), it->first.c_str(),
                percentile(v, 0.5), percentile(v, 0.1), percentile(v, 0.9));
    }
    fclose(f);
}
