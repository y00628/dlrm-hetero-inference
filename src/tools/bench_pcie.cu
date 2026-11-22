#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
    size_t idx = blockDim.x * (size_t)blockIdx.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

int main(int argc, char** argv) {
    size_t mb = (argc > 1) ? atoi(argv[1]) : 256;
    if (mb <= 0) mb = 256;
    size_t bytes = mb * 1024 * 1024;
    size_t n_float4 = bytes / sizeof(float4);
    int iters = 20;

    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("======================================================================\n");
    printf("PCIe & Memory Bandwidth Microbenchmark\n");
    printf("Device: %s (SM %d.%d, %.1f GB VRAM)\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);
    printf("Buffer size: %zu MB (%d iterations per test)\n", mb, iters);
    printf("======================================================================\n\n");

    float4 *d_src = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemset(d_src, 1, bytes));

    int threads = 256;
    int blocks = (int)((n_float4 + threads - 1) / threads);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 1. GPU VRAM Bandwidth (pure device copy)
    copy_kernel<<<blocks, threads>>>(d_src, d_dst, n_float4); // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        copy_kernel<<<blocks, threads>>>(d_src, d_dst, n_float4);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_vram = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_vram, start, stop));
    double bw_vram = (2.0 * bytes * iters) / (ms_vram * 1e6); // 1 read + 1 write

    // 2. Pageable H2D Bandwidth (standard malloc)
    void* h_pageable = malloc(bytes);
    memset(h_pageable, 1, bytes);
    CUDA_CHECK(cudaMemcpy(d_dst, h_pageable, bytes, cudaMemcpyHostToDevice)); // warmup
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaMemcpy(d_dst, h_pageable, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_pageable = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pageable, start, stop));
    double bw_pageable = (bytes * iters) / (ms_pageable * 1e6);
    free(h_pageable);

    // 3. Pinned H2D Bandwidth (cudaMallocHost)
    void* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
    memset(h_pinned, 1, bytes);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_pinned, bytes, cudaMemcpyHostToDevice, stream)); // warmup
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_dst, h_pinned, bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ms_pinned = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pinned, start, stop));
    double bw_pinned = (bytes * iters) / (ms_pinned * 1e6);
    CUDA_CHECK(cudaFreeHost(h_pinned));

    // 4. Zero-Copy Host Read Bandwidth (cudaHostAllocMapped)
    void* h_zero = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_zero, bytes, cudaHostAllocMapped));
    memset(h_zero, 1, bytes);
    float4* d_zero_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_zero_ptr, h_zero, 0));
    copy_kernel<<<blocks, threads, 0, stream>>>(d_zero_ptr, d_dst, n_float4); // warmup
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        copy_kernel<<<blocks, threads, 0, stream>>>(d_zero_ptr, d_dst, n_float4);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ms_zero = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_zero, start, stop));
    double bw_zero = (bytes * iters) / (ms_zero * 1e6);
    CUDA_CHECK(cudaFreeHost(h_zero));

    printf("| Transfer Mode                        | Measured Bandwidth | Latency per %zu MB |\n", mb);
    printf("|:-------------------------------------|-------------------:|--------------------:|\n");
    printf("| GPU VRAM Read/Write (Coalesced)      | %15.2f GB/s | %15.2f ms |\n", bw_vram, ms_vram / iters);
    printf("| PCIe Pageable Host-to-Device (H2D)   | %15.2f GB/s | %15.2f ms |\n", bw_pageable, ms_pageable / iters);
    printf("| PCIe Pinned Host-to-Device (H2D)     | %15.2f GB/s | %15.2f ms |\n", bw_pinned, ms_pinned / iters);
    printf("| PCIe Zero-Copy GPU Kernel Read       | %15.2f GB/s | %15.2f ms |\n", bw_zero, ms_zero / iters);
    printf("======================================================================\n");
    printf("Ratio r = BW_VRAM / BW_PCIe_Pinned: %.2fx\n", bw_vram / bw_pinned);
    printf("======================================================================\n");

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

