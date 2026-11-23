// Hot-row profiler: ranks every embedding row by how often a trace looks it up, writes the ranking as the
// "hot file" that gpu_tier_pool and gpu_tier_zerocopy read, and prints the cache hit rate per cache size.
//
// Usage: ./bin/hot_profile <config> <hot_file> [--profile-frac F] [--fractions f1,f2,...] [--pad F]
//   --profile-frac  share of the trace's batches used for counting (default 0.5). The remaining batches are
//                   a hold-out set: the hit rate on them says how well the ranking carries over to unseen
//                   traffic. With one batch there is no hold-out. Use 1.0 to count over everything.
//   --fractions     cache sizes to report, as a fraction of all rows (default 0.01,0.05,0.1,0.25,0.5)
//   --pad           make the file hold at least this fraction of all rows (default: the largest of
//                   --fractions). Rows never seen while counting are appended in table/row order so that a
//                   cache of that size is always full. Use 0 to write only the rows that were seen.
//
// Hot file format (read by gpu_tier_pool): int32 pairs (table, row), hottest first, no header. A cache of
// fraction f holds the first floor(f * total_rows) pairs. Ties in the count are broken by lower global row.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "config.h"
#include "trace.h"

using namespace std;
using i64 = long long;

static vector<double> parse_list(const string& s) {
    vector<double> v;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == string::npos) comma = s.size();
        v.push_back(atof(s.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <config> <hot_file> [--profile-frac F] [--fractions f1,f2,...] [--pad F]\n", argv[0]);
        return 1;
    }
    Config cfg = load_config(argv[1]);
    const char* out_path = argv[2];
    double profile_frac = 0.5;
    vector<double> fractions = {0.01, 0.05, 0.1, 0.25, 0.5};
    double pad = -1;  // -1 = largest fraction
    for (int a = 3; a < argc; a++) {
        string arg = argv[a];
        if (arg == "--profile-frac" && a + 1 < argc) profile_frac = atof(argv[++a]);
        else if (arg == "--fractions" && a + 1 < argc) fractions = parse_list(argv[++a]);
        else if (arg == "--pad" && a + 1 < argc) pad = atof(argv[++a]);
        else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[a]);
            return 1;
        }
    }
    if (fractions.empty() || profile_frac <= 0 || profile_frac > 1) {
        fprintf(stderr, "bad arguments (profile-frac must be in (0, 1], fractions must not be empty)\n");
        return 1;
    }
    if (pad < 0) pad = *max_element(fractions.begin(), fractions.end());

    int M = cfg.num_tables;
    vector<i64> base(M + 1, 0);
    for (int t = 0; t < M; t++) base[t + 1] = base[t] + cfg.rows[t];
    i64 total_rows = base[M];
    if (total_rows >= INT_MAX) {
        fprintf(stderr, "total rows %lld does not fit in int32\n", total_rows);
        return 1;
    }

    Trace tr = load_trace(cfg.trace_file);
    if (tr.num_tables != M) {
        fprintf(stderr, "trace has %d tables, config has %d\n", tr.num_tables, M);
        return 1;
    }
    int nb = (int)tr.batches.size();
    int profile_batches = max(1, (int)(profile_frac * nb + 0.5));
    profile_batches = min(profile_batches, nb);
    int holdout_batches = nb - profile_batches;

    // Count lookups per global row over the profiling batches
    vector<uint32_t> count(total_rows, 0);
    i64 profile_lookups = 0, holdout_lookups = 0;
    for (int b = 0; b < nb; b++) {
        for (int t = 0; t < M; t++) {
            const auto& idx = tr.batches[b].tables[t].indices;
            if (b < profile_batches) {
                for (int r : idx) count[base[t] + r]++;
                profile_lookups += (i64)idx.size();
            } else {
                holdout_lookups += (i64)idx.size();
            }
        }
    }

    // Order: count descending, then global row ascending
    vector<int> order;
    for (i64 g = 0; g < total_rows; g++) {
        if (count[g] > 0) order.push_back((int)g);
    }
    i64 seen_rows = (i64)order.size();
    sort(order.begin(), order.end(), [&](int a, int b) {
        return count[a] != count[b] ? count[a] > count[b] : a < b;
    });
    // Pad with unseen rows (in row order) so a cache of size `pad` is always full
    i64 want = min((i64)(pad * total_rows), total_rows);
    if ((i64)order.size() < want) {
        vector<char> seen(total_rows, 0);
        for (int g : order) seen[g] = 1;
        for (i64 g = 0; g < total_rows && (i64)order.size() < want; g++) {
            if (!seen[g]) order.push_back((int)g);
        }
    }

    // Write the hot file
    FILE* f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    vector<int> pairs;
    pairs.reserve(order.size() * 2);
    for (int g : order) {
        // table of global row g: order is not sorted by table, so search
        int t = (int)(upper_bound(base.begin(), base.end(), (i64)g) - base.begin()) - 1;
        pairs.push_back(t);
        pairs.push_back((int)(g - base[t]));
    }
    fwrite(pairs.data(), sizeof(int), pairs.size(), f);
    fclose(f);

    // rank[g] = position in the file, INT_MAX if not listed
    vector<int> rank(total_rows, INT_MAX);
    for (size_t i = 0; i < order.size(); i++) rank[order[i]] = (int)i;

    printf("config %s: M=%d total rows=%lld (%.3f GB at D=%d)\n", cfg.name.c_str(), M, total_rows,
           total_rows * cfg.emb_dim * 4.0 / 1e9, cfg.emb_dim);
    printf("trace: %d batches; profiled on the first %d (%lld lookups), hold-out %d batches (%lld lookups)\n", nb,
           profile_batches, profile_lookups, holdout_batches, holdout_lookups);
    printf("rows seen while profiling: %lld of %lld (%.1f%%); wrote %zu entries to %s\n", seen_rows, total_rows,
           100.0 * seen_rows / total_rows, order.size(), out_path);
    printf("\n%10s %12s %9s %12s %12s\n", "cache", "rows cached", "GB", "hit in-sample", "hit hold-out");

    for (double frac : fractions) {
        i64 capacity = min((i64)(frac * total_rows), (i64)order.size());
        i64 hit_in = 0, hit_out = 0;
        for (int b = 0; b < nb; b++) {
            for (int tb = 0; tb < M; tb++) {
                for (int r : tr.batches[b].tables[tb].indices) {
                    if (rank[base[tb] + r] < capacity) (b < profile_batches ? hit_in : hit_out)++;
                }
            }
        }
        char out_buf[32];
        if (holdout_lookups > 0) snprintf(out_buf, sizeof out_buf, "%.3f", (double)hit_out / holdout_lookups);
        else snprintf(out_buf, sizeof out_buf, "n/a");
        printf("%9.0f%% %12lld %9.3f %12.3f %12s\n", frac * 100, capacity, capacity * cfg.emb_dim * 4.0 / 1e9,
               (double)hit_in / profile_lookups, out_buf);
    }
    printf("\n(a cache larger than the file holds only the listed rows, same as gpu_tier_pool would load it)\n");
    return 0;
}
