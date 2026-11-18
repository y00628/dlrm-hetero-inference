# GHC (RTX 2080): make
# PSC (V100):     make ARCH=sm_70

CXX = g++
NVCC = nvcc
ARCH = sm_75

CXXFLAGS = -O3 -std=c++17 -Wall -fopenmp -mavx2 -mfma -Iinclude
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Iinclude -Xcompiler -fopenmp,-mavx2,-mfma

COMMON = src/common/config.cpp src/common/trace.cpp src/common/timing.cpp src/common/check.cpp
HEADERS = $(wildcard include/*.h)

all: bin/trace_info bin/check

bin/trace_info: src/tools/trace_info.cpp $(COMMON) $(HEADERS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/tools/trace_info.cpp $(COMMON) -o $@

bin/check: src/tools/check.cpp $(COMMON) $(HEADERS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/tools/check.cpp $(COMMON) -o $@

clean:
	rm -rf bin

.PHONY: all clean
