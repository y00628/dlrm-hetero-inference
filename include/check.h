#pragma once

#include <string>
#include <vector>

// Correctness dumps are raw float32 arrays. Each implementation writes, for batches 0 and 1:
//   <name>_b<i>_pooled.bin   pooled embeddings [B][M][D]
//   <name>_b<i>_ctr.bin      CTR scores [B]
void write_dump(const std::string& path, const float* data, size_t count);
std::vector<float> read_dump(const std::string& path);

// Counts elements that fail |x - ref| <= tol * max(|ref|, 1) (NaN also fails).
// Writes the largest error to *max_err.
long long count_mismatches(const std::vector<float>& ref, const std::vector<float>& x, double tol,
                           double* max_err);
