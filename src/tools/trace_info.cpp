// Loads a config and its trace, checks that they match, and prints pooling and skew.
// Usage: ./bin/trace_info configs/small.cfg

#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>

#include "config.h"
#include "timing.h"
#include "trace.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: %s <config>\n", argv[0]);
        return 1;
    }
    Config cfg = load_config(argv[1]);

    double t0 = now_ms();
    Trace tr = load_trace(cfg.trace_file);
    printf("loaded %s in %.0f ms\n", cfg.trace_file.c_str(), now_ms() - t0);

    if (tr.num_tables != cfg.num_tables || tr.batch_size != cfg.batch_size || tr.num_dense != cfg.num_dense) {
        printf("FAIL: trace header does not match config\n");
        return 1;
    }

    double table_gb = 0;
    for (int t = 0; t < cfg.num_tables; t++) table_gb += cfg.rows[t] * cfg.emb_dim * 4.0 / 1e9;
    printf("M=%d D=%d B=%d batches=%zu tables=%.2f GB interaction_dim=%d\n", cfg.num_tables, cfg.emb_dim,
           cfg.batch_size, tr.batches.size(), table_gb, interaction_dim(cfg));

    // Mean pooling factor per table, and indices in range
    for (int t = 0; t < cfg.num_tables; t++) {
        long long total = 0;
        for (const Batch& batch : tr.batches) {
            for (int idx : batch.tables[t].indices) {
                if (idx < 0 || idx >= cfg.rows[t]) {
                    printf("FAIL: table %d has index %d out of range\n", t, idx);
                    return 1;
                }
            }
            total += batch.tables[t].indices.size();
        }
        printf("table %d: rows=%lld mean L=%.2f\n", t, cfg.rows[t],
               (double)total / (tr.batches.size() * cfg.batch_size));
    }

    // Skew of table 0: share of lookups that go to the hottest 1% of rows
    std::vector<int> counts(cfg.rows[0], 0);
    long long lookups = 0;
    for (const Batch& batch : tr.batches) {
        for (int idx : batch.tables[0].indices) counts[idx]++;
        lookups += batch.tables[0].indices.size();
    }
    std::sort(counts.begin(), counts.end(), std::greater<int>());
    long long hot = 0;
    for (size_t i = 0; i < std::max<size_t>(1, counts.size() / 100); i++) hot += counts[i];
    printf("table 0, alpha=%.2f: hottest 1%% of rows get %.1f%% of lookups\n", cfg.alpha, 100.0 * hot / lookups);

    printf("OK\n");
    return 0;
}
