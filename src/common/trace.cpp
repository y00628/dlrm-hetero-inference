#include "trace.h"

#include <cstdio>
#include <cstdlib>

static void read_or_die(void* dst, size_t elem_size, size_t count, FILE* f) {
    if (fread(dst, elem_size, count, f) != count) {
        fprintf(stderr, "trace file is truncated\n");
        exit(1);
    }
}

Trace load_trace(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open trace %s\n", path.c_str());
        exit(1);
    }

    int header[4];  // num_tables, batch_size, num_batches, num_dense
    read_or_die(header, sizeof(int), 4, f);

    Trace tr;
    tr.num_tables = header[0];
    tr.batch_size = header[1];
    tr.num_dense = header[3];
    tr.batches.resize(header[2]);

    int B = tr.batch_size;
    for (Batch& batch : tr.batches) {
        batch.dense.resize((size_t)B * tr.num_dense);
        read_or_die(batch.dense.data(), sizeof(float), batch.dense.size(), f);

        batch.tables.resize(tr.num_tables);
        for (TableBatch& tb : batch.tables) {
            tb.offsets.resize(B + 1);
            read_or_die(tb.offsets.data(), sizeof(int), B + 1, f);
            tb.indices.resize(tb.offsets[B]);
            read_or_die(tb.indices.data(), sizeof(int), tb.indices.size(), f);
        }
    }
    fclose(f);
    return tr;
}
