// Multi-threaded CPU reference. This is the numerical ground truth for every other implementation.
// Usage: ./bin/cpu_ref configs/tiny.cfg [results.csv]
//
// Writes ref_b0_pooled.bin, ref_b0_ctr.bin, ref_b1_pooled.bin, ref_b1_ctr.bin (see include/check.h)
// and appends per-stage times to the CSV (default results.csv).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "check.h"
#include "config.h"
#include "init.h"
#include "timing.h"
#include "trace.h"

// ---------------------------------------------------------------------------------------------
// Vector helpers (AVX2 when compiled with -mavx2, scalar otherwise)
// ---------------------------------------------------------------------------------------------

static inline float dot(const float* a, const float* b, int n) {
    int i = 0;
    float sum = 0.0f;
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    for (int k = 0; k < 8; k++) {
        sum += tmp[k];
    }
#endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// dst[0..n) += src[0..n)
static inline void add_to(float* dst, const float* src, int n) {
    int i = 0;
#ifdef __AVX2__
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
#endif
    for (; i < n; i++) 
        dst[i] += src[i];
}

// ---------------------------------------------------------------------------------------------
// Model pieces
// ---------------------------------------------------------------------------------------------

struct Layer {
    int in, out;
    std::vector<float> W;  // [out][in]
};

static std::vector<Layer> make_mlp(uint64_t seed, const std::vector<int>& dims, int layer_id_base) {
    std::vector<Layer> layers;
    for (size_t i = 0; i + 1 < dims.size(); i++) {
        Layer L;
        L.in = dims[i];
        L.out = dims[i + 1];
        L.W.resize((size_t)L.in * L.out);
        fill_mlp_layer(seed, layer_id_base + (int)i, L.out, L.in, L.W.data());
        layers.push_back(std::move(L));
    }
    return layers;
}

// X: [B][in] -> Y: [B][out],  y = W x for each row, optional relu.
// Parallel over samples; each thread streams the whole W (small enough to stay in L2).
static void linear(const Layer& L, const float* X, float* Y, int B, bool relu) {
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const float* x = X + (size_t)b * L.in;
        float* y = Y + (size_t)b * L.out;
        for (int o = 0; o < L.out; o++) {
            float v = dot(x, L.W.data() + (size_t)o * L.in, L.in);
            y[o] = relu ? std::max(v, 0.0f) : v;
        }
    }
}

// Runs an MLP. Returns the final activations [B][last_out] in `out`.
// relu_last = false for the top MLP (its last layer feeds the sigmoid).
static void run_mlp(const std::vector<Layer>& layers, const float* X, std::vector<float>& out, int B,
                    bool relu_last) {
    std::vector<float> cur, next;
    const float* in = X;
    for (size_t i = 0; i < layers.size(); i++) {
        next.assign((size_t)B * layers[i].out, 0.0f);
        bool relu = (i + 1 < layers.size()) || relu_last;
        linear(layers[i], in, next.data(), B, relu);
        cur.swap(next);
        in = cur.data();
    }
    out = cur;
}

// pooled[b][t][:] = sum of the rows of table t in bag b.   pooled: [B][M][D]
static void pool_embeddings(const Config& cfg, const std::vector<std::vector<float>>& tables, const Batch& batch,
                            float* pooled) {
    const int B = cfg.batch_size, M = cfg.num_tables, D = cfg.emb_dim;
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < M; t++) {
            float* dst = pooled + ((size_t)b * M + t) * D;
            std::fill(dst, dst + D, 0.0f);
            const TableBatch& tb = batch.tables[t];
            for (int k = tb.offsets[b]; k < tb.offsets[b + 1]; k++) {
                add_to(dst, tables[t].data() + (size_t)tb.indices[k] * D, D);
            }
        }
    }
}

// z[b] = [bot[b], dot(v_i, v_j) for i < j],  v = (bot[b], e_0[b], ..., e_{M-1}[b])
static void interact(const Config& cfg, const float* bot, const float* pooled, float* z) {
    const int B = cfg.batch_size, M = cfg.num_tables, D = cfg.emb_dim;
    const int n = M + 1, Z = interaction_dim(cfg);
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const float* v[64];  // pointers to the n vectors (n <= 64 is checked in main)
        v[0] = bot + (size_t)b * D;
        for (int t = 0; t < M; t++) v[t + 1] = pooled + ((size_t)b * M + t) * D;

        float* out = z + (size_t)b * Z;
        std::copy(v[0], v[0] + D, out);
        int k = D;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) out[k++] = dot(v[i], v[j], D);
    }
}

// ---------------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <config> [results.csv]\n", argv[0]);
        return 1;
    }
    Config cfg = load_config(argv[1]);
    std::string csv = argc > 2 ? argv[2] : "results.csv";

    const int B = cfg.batch_size, M = cfg.num_tables, D = cfg.emb_dim;
    if (cfg.mlp_bot.front() != cfg.num_dense || cfg.mlp_bot.back() != D ||
        cfg.mlp_top.front() != interaction_dim(cfg) || cfg.mlp_top.back() != 1 || M + 1 > 64) {
        fprintf(stderr, "config dimensions are inconsistent (mlp_bot, mlp_top, emb_dim, num_tables)\n");
        return 1;
    }

    Trace trace = load_trace(cfg.trace_file);
    if (trace.num_tables != M || trace.batch_size != B || trace.num_dense != cfg.num_dense) {
        fprintf(stderr, "trace does not match config\n");
        return 1;
    }
    const int nb = (int)trace.batches.size();

    // Parameters, generated from the seed (include/init.h)
    double t0 = now_ms();
    std::vector<std::vector<float>> tables(M);
    for (int t = 0; t < M; t++) {
        tables[t].resize((size_t)cfg.rows[t] * D);
        fill_table(cfg.seed, t, 0, cfg.rows[t], D, tables[t].data());
    }
    std::vector<Layer> bot_mlp = make_mlp(cfg.seed, cfg.mlp_bot, 0);
    std::vector<Layer> top_mlp = make_mlp(cfg.seed, cfg.mlp_top, 100);
    printf("parameters ready in %.0f ms\n", now_ms() - t0);

    const int Z = interaction_dim(cfg);
    std::vector<float> pooled((size_t)B * M * D), z((size_t)B * Z), bot, top, ctr(B);
    StageTimes times;

    // Batches 0 and 1 are the warm-up batches that get dumped; the trace is reused cyclically.
    const int total_iters = WARMUP_BATCHES + TIMED_BATCHES;
    for (int it = 0; it < total_iters; it++) {
        const Batch& batch = trace.batches[it % nb];

        double a = now_ms();
        run_mlp(bot_mlp, batch.dense.data(), bot, B, /*relu_last=*/true);
        double b_ = now_ms();
        pool_embeddings(cfg, tables, batch, pooled.data());
        double c = now_ms();
        interact(cfg, bot.data(), pooled.data(), z.data());
        run_mlp(top_mlp, z.data(), top, B, /*relu_last=*/false);
        for (int i = 0; i < B; i++) ctr[i] = 1.0f / (1.0f + std::exp(-top[i]));
        double d = now_ms();

        if (it < 2) {
            std::string p = "ref_b" + std::to_string(it);
            write_dump(p + "_pooled.bin", pooled.data(), pooled.size());
            write_dump(p + "_ctr.bin", ctr.data(), ctr.size());
        }
        if (it >= WARMUP_BATCHES) {
            times["dense_bot"].push_back(b_ - a);
            times["lookup"].push_back(c - b_);
            times["interact_top"].push_back(d - c);
            times["total"].push_back(d - a);
        }
    }

    std::vector<double> tot = times["total"];
    std::sort(tot.begin(), tot.end());
    printf("%s: median %.2f ms/batch (B=%d) -> %.0f samples/s\n", cfg.name.c_str(), tot[tot.size() / 2], B,
           B / (tot[tot.size() / 2] / 1000.0));
    append_results(csv, cfg, "cpu_ref", 0, times);
    return 0;
}
