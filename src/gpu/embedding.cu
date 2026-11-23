#include "embedding.h"

#include "check.h"
#include "cuda_utils.h"

using namespace std;
using i64 = long long;

vector<i64> table_base(const Config& cfg) {
    vector<i64> base(cfg.num_tables + 1, 0);
    for (int t = 0; t < cfg.num_tables; t++) base[t + 1] = base[t] + cfg.rows[t];
    return base;
}

vector<FlatBatch> flatten_trace(Trace& tr, const vector<i64>& base) {
    int M = tr.num_tables;
    int B = tr.batch_size;
    vector<FlatBatch> out;
    out.reserve(tr.batches.size());

    for (auto& batch : tr.batches) {
        FlatBatch fb;
        fb.num_bags = B * M;
        fb.num_lookups = 0;
        for (int t = 0; t < M; t++) fb.num_lookups += (int)batch.tables[t].indices.size();

        fb.offsets = (int*)malloc((fb.num_bags + 1) * sizeof(int));
        fb.rows = (int*)malloc(fb.num_lookups * sizeof(int));
        if (!fb.offsets || !fb.rows) {
            fprintf(stderr, "failed to allocate host memory for FlatBatch\n");
            exit(1);
        }

        int pos = 0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < M; t++) {
                const auto& tb = batch.tables[t];
                fb.offsets[b * M + t] = pos;
                for (int i = tb.offsets[b]; i < tb.offsets[b + 1]; i++) {
                    fb.rows[pos] = (int)(base[t] + tb.indices[i]);
                    pos++;
                }
            }
        }
        fb.offsets[fb.num_bags] = pos;
        out.push_back(fb);

        // Release batch data immediately to reduce peak memory
        vector<float>().swap(batch.dense);
        vector<TableBatch>().swap(batch.tables);
    }
    tr.batches.clear();
    return out;
}

void free_flat_batches(vector<FlatBatch>& batches) {
    for (auto& fb : batches) {
        free(fb.offsets);
        free(fb.rows);
    }
    batches.clear();
}

__global__ void pool_kernel(const float* table, const int* offsets, const int* rows, int num_bags, int D,
                            const float* init, float* out) {
    int bag = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (bag >= num_bags) return;

    int start = offsets[bag];
    int end = offsets[bag + 1];
    for (int d = lane; d < D; d += 32) {
        float sum = (init != nullptr) ? init[(i64)bag * D + d] : 0.0f;
        for (int i = start; i < end; i++) {
            sum += table[(i64)rows[i] * D + d];
        }
        out[(i64)bag * D + d] = sum;
    }
}

void launch_pool(const float* table, const int* offsets, const int* rows, int num_bags, int D,
                 const float* init, float* out, cudaStream_t stream) {
    int threads = 256;  // 8 warps = 8 bags per block
    int blocks = (num_bags * 32 + threads - 1) / threads;
    pool_kernel<<<blocks, threads, 0, stream>>>(table, offsets, rows, num_bags, D, init, out);
    CUDA_CHECK(cudaGetLastError());
}

void dump_device(const float* d_data, size_t count, const string& path) {
    vector<float> h(count);
    CUDA_CHECK(cudaMemcpy(h.data(), d_data, count * sizeof(float), cudaMemcpyDeviceToHost));
    write_dump(path, h.data(), count);
}
