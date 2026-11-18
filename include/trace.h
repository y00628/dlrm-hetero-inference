#pragma once

#include <string>
#include <vector>

// Trace file (binary, little-endian), written by tools/gen_trace.py and the data converters:
//   header:    int32 num_tables, batch_size, num_batches, num_dense
//   per batch: float32 dense[B][num_dense]
//              for each table: int32 offsets[B+1], int32 indices[offsets[B]]
// Bag b of a table is indices[offsets[b] .. offsets[b+1]).

struct TableBatch {
    std::vector<int> offsets;  // B + 1
    std::vector<int> indices;
};

struct Batch {
    std::vector<float> dense;        // [B][num_dense]
    std::vector<TableBatch> tables;  // one per table
};

struct Trace {
    int num_tables;
    int batch_size;
    int num_dense;
    std::vector<Batch> batches;
};

// Loads the whole trace into memory.
Trace load_trace(const std::string& path);
