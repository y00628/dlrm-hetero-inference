// Compares an output dump against the CPU reference dump.
// Usage: ./bin/check <reference.bin> <output.bin> [tolerance, default 1e-4]
// Exit code 0 = pass, 1 = fail.

#include <cstdio>
#include <cstdlib>

#include "check.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <reference.bin> <output.bin> [tolerance]\n", argv[0]);
        return 1;
    }
    double tol = (argc > 3) ? atof(argv[3]) : 1e-4;
    std::vector<float> ref = read_dump(argv[1]);
    std::vector<float> out = read_dump(argv[2]);
    if (ref.size() != out.size()) {
        printf("FAIL: reference has %zu values, output has %zu\n", ref.size(), out.size());
        return 1;
    }

    double max_err;
    long long bad = count_mismatches(ref, out, tol, &max_err);
    printf("%s: %lld of %zu values outside tolerance %g, max error %g\n", bad == 0 ? "PASS" : "FAIL", bad,
           ref.size(), tol, max_err);
    return bad == 0 ? 0 : 1;
}
