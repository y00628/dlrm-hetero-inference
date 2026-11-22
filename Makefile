# GHC (RTX 2080): make
# PSC (V100):     make ARCH=sm_70

CXX = g++
NVCC = nvcc
ARCH = sm_75

CXXFLAGS = -O3 -std=c++17 -Wall -fopenmp -mavx2 -mfma -Iinclude
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -allow-unsupported-compiler -Iinclude -Xcompiler -fopenmp,-mavx2,-mfma
NVLIBS = -lgomp

COMMON = src/common/config.cpp src/common/trace.cpp src/common/timing.cpp src/common/check.cpp
GPU_COMMON = src/gpu/embedding.cu
HEADERS = $(wildcard include/*.h)

all: bin/trace_info bin/check bin/cpu_ref bin/gpu_single bin/gpu_tier_pool

bin/cpu_ref: src/ref/cpu_ref.cpp $(COMMON) $(HEADERS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/ref/cpu_ref.cpp $(COMMON) -o $@

bin/trace_info: src/tools/trace_info.cpp $(COMMON) $(HEADERS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/tools/trace_info.cpp $(COMMON) -o $@

bin/check: src/tools/check.cpp $(COMMON) $(HEADERS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/tools/check.cpp $(COMMON) -o $@

bin/gpu_single: src/gpu/gpu_single.cu $(GPU_COMMON) $(COMMON) $(HEADERS)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) src/gpu/gpu_single.cu $(GPU_COMMON) $(COMMON) -o $@ $(NVLIBS)

bin/gpu_tier_pool: src/gpu/gpu_tier_pool.cu src/cpu/cold_pool.cpp $(GPU_COMMON) $(COMMON) $(HEADERS)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) src/gpu/gpu_tier_pool.cu src/cpu/cold_pool.cpp $(GPU_COMMON) $(COMMON) -o $@ $(NVLIBS)

clean:
	rm -rf bin

.PHONY: all clean
