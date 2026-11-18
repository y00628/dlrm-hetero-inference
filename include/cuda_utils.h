#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err = (call);                                                              \
        if (err != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

// Times GPU work on one stream with two CUDA events.
struct GpuTimer {
    cudaEvent_t start_ev, stop_ev;

    GpuTimer() {
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }
    void start(cudaStream_t s = 0) { cudaEventRecord(start_ev, s); }
    void stop(cudaStream_t s = 0) { cudaEventRecord(stop_ev, s); }
    float elapsed_ms() {
        cudaEventSynchronize(stop_ev);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        return ms;
    }
};
