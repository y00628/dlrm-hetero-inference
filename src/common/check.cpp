#include "check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

void write_dump(const std::string& path, const float* data, size_t count) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    fwrite(data, sizeof(float), count, f);
    fclose(f);
}

std::vector<float> read_dump(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> data(bytes / sizeof(float));
    size_t got = fread(data.data(), sizeof(float), data.size(), f);
    fclose(f);
    data.resize(got);
    return data;
}

long long count_mismatches(const std::vector<float>& ref, const std::vector<float>& x, double tol,
                           double* max_err) {
    long long bad = 0;
    *max_err = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        double err = std::fabs((double)x[i] - ref[i]) / std::max(std::fabs((double)ref[i]), 1.0);
        if (!(err <= tol)) bad++;  // also counts NaN
        if (err > *max_err || std::isnan(err)) *max_err = err;
    }
    return bad;
}
