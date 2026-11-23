#!/usr/bin/env bash
# Automated Experiment Runner for Phase 1 Deliverables:
# Part B: Skew (alpha) x Cache fraction sweep on small.cfg (with hot file)
# Part C: Out-of-Core memory tiering on medium.cfg (10 GB model on 8.3 GB GPU)

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -d /usr/local/cuda/bin ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

MODE="${1:-all}"  # "b", "c", or "all"
PY="${PY:-python3}"

echo "Building GPU binaries..."
make gpu

mkdir -p data results/gpu

# -----------------------------------------------------------------------------
# Part B: Skew x Cache fraction sweep on small.cfg
# -----------------------------------------------------------------------------
run_part_b() {
    echo "=========================================================================="
    echo "Starting Part B: Skew (alpha) x Cache Size Sweep on small.cfg"
    echo "=========================================================================="
    local res_csv="results/gpu/results_part_b.csv"
    rm -f "$res_csv"

    local alphas=(0 0.6 1.0 1.2)
    local caches=(0.0 0.01 0.05 0.1 0.25 0.5 1.0)

    for a in "${alphas[@]}"; do
        local cfg="data/small_a${a}.cfg"
        local tr_main="data/small_a${a}.trace"
        local tr_prof="data/small_a${a}_prof.trace"
        local hot_bin="data/small_a${a}_hot.bin"

        echo "--- [Part B] Preparing Alpha = $a ---"
        # Generate temporary config for this alpha
        sed -E "s/^zipf_alpha[[:space:]]*=.*/zipf_alpha = $a/; s|^trace_file[[:space:]]*=.*|trace_file = $tr_main|" configs/small.cfg > "$cfg"

        # 1. Generate main trace (if missing)
        if [ ! -f "$tr_main" ]; then
            echo "  Generating main trace: $tr_main"
            "$PY" tools/gen_trace.py "$cfg" --out "$tr_main"
        fi

        # 2. Generate profiling trace with different seed
        if [ ! -f "$tr_prof" ]; then
            echo "  Generating profiling trace: $tr_prof"
            "$PY" tools/gen_trace.py "$cfg" --seed 1001 --out "$tr_prof"
        fi

        # 3. Generate hot file from profiling trace
        if [ ! -f "$hot_bin" ]; then
            echo "  Profiling hot rows -> $hot_bin"
            "$PY" tools/profile_hot.py "$tr_prof" --out "$hot_bin"
        fi

        # 4. Sweep cache fractions
        for c in "${caches[@]}"; do
            echo "  Running: alpha=$a, cache=$c"
            ./bin/gpu_tier_pool "$cfg" "$c" "$hot_bin" "$res_csv"
        done
    done
    echo "Part B completed! Results saved to $res_csv"
}

# -----------------------------------------------------------------------------
# Part C: Out-of-Core Tiering on medium.cfg (10 GB table)
# -----------------------------------------------------------------------------
run_part_c() {
    echo "=========================================================================="
    echo "Starting Part C: Out-of-Core Memory Tiering on medium.cfg (10 GB)"
    echo "=========================================================================="
    local res_csv="results/gpu/results_part_c.csv"
    rm -f "$res_csv"

    echo "Checking Host Memory:"
    free -h || true

    # Step 1: Prove gpu_single fails with OOM
    echo "--- Testing gpu_single on medium.cfg (Expected: OOM) ---"
    # Ensure baseline medium trace exists
    if [ ! -f "data/medium.trace" ]; then
        "$PY" tools/gen_trace.py configs/medium.cfg
    fi
    set +e
    ./bin/gpu_single configs/medium.cfg "$res_csv" 2>&1 | tee results/gpu/medium_oom_log.txt
    local ret=$?
    set -e
    if [ $ret -ne 0 ]; then
        echo ">>> Verified: gpu_single crashed with OOM as expected! Logged to results/gpu/medium_oom_log.txt"
    else
        echo ">>> Note: gpu_single unexpectedly succeeded (GPU VRAM > 10GB?)"
    fi

    # Step 2: Test gpu_tier_pool across alphas and caches
    local alphas=(0.6 1.0 1.2)
    local caches=(0.05 0.1 0.25 0.5)

    for a in "${alphas[@]}"; do
        local cfg="data/medium_a${a}.cfg"
        local tr_main="data/medium_a${a}.trace"
        local tr_prof="data/medium_a${a}_prof.trace"
        local hot_bin="data/medium_a${a}_hot.bin"

        echo "--- [Part C] Preparing Alpha = $a ---"
        sed -E "s/^zipf_alpha[[:space:]]*=.*/zipf_alpha = $a/; s|^trace_file[[:space:]]*=.*|trace_file = $tr_main|" configs/medium.cfg > "$cfg"

        if [ ! -f "$tr_main" ]; then
            echo "  Generating main trace: $tr_main"
            "$PY" tools/gen_trace.py "$cfg" --out "$tr_main"
        fi
        if [ ! -f "$tr_prof" ]; then
            echo "  Generating profiling trace: $tr_prof"
            "$PY" tools/gen_trace.py "$cfg" --seed 1001 --out "$tr_prof"
        fi
        if [ ! -f "$hot_bin" ]; then
            echo "  Profiling hot rows -> $hot_bin"
            "$PY" tools/profile_hot.py "$tr_prof" --out "$hot_bin"
        fi

        for c in "${caches[@]}"; do
            echo "  Running medium: alpha=$a, cache=$c"
            ./bin/gpu_tier_pool "$cfg" "$c" "$hot_bin" "$res_csv"
        done
    done
    echo "Part C completed! Results saved to $res_csv"
}

case "$MODE" in
    b|B)
        run_part_b
        ;;
    c|C)
        run_part_c
        ;;
    all|ALL)
        run_part_b
        run_part_c
        ;;
    *)
        echo "Usage: $0 [b|c|all]"
        exit 1
        ;;
esac
