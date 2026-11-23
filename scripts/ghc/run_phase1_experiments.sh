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

# GHC sets a 16 GiB virtual-memory limit per process (ulimit -v). CUDA maps GPU allocations
# into the process address space, so 10 GB host table + GPU cache can exceed it.
# Raise the soft limit up to the hard limit (no effect if the hard limit is also 16 GiB).
ulimit -v "$(ulimit -Hv)" 2>/dev/null || true
echo "virtual memory limit: $(ulimit -v) KB"
PY="${PY:-python3}"

echo "Building GPU binaries..."
make gpu

mkdir -p data results/gpu

# Every invocation has separate logs; aggregate CSVs are append-only.
RUN_DIR=$(mktemp -d results/gpu/phase1_run_$(date +%Y%m%d_%H%M%S)_XXXXXX)
printf 'label,exit_code,status,log\n' > "$RUN_DIR/status.csv"
failures=0

diagnostics() {
    echo "--- diagnostics $(date -u) ---"
    echo "virtual memory soft/hard (KB): $(ulimit -Sv) / $(ulimit -Hv)"
    echo "locked memory soft/hard (KB): $(ulimit -Sl) / $(ulimit -Hl)"
    free -h || true
    nvidia-smi || true
    cat /proc/$$/limits 2>/dev/null || true
}
diagnostics > "$RUN_DIR/environment.log" 2>&1
echo "Run logs: $RUN_DIR (existing CSVs preserved; repeated points are appended)"

# Conditional pipeline avoids errexit; inspect both producer and tee exit codes.
run_logged() {
    local label="$1" expected="$2"
    shift 2
    local log="$RUN_DIR/$label.log" rc status
    local codes
    if "$@" 2>&1 | tee "$log"; then
        codes=("${PIPESTATUS[@]}")
    else
        codes=("${PIPESTATUS[@]}")
    fi
    rc=${codes[0]}
    status=PASS
    if [ "${codes[1]}" -ne 0 ]; then
        status=LOG_FAILED
    elif [ "$rc" -ne 0 ]; then
        status=FAILED
        if grep -Eiq '(CUDA error.*out of memory|CUDA cache allocation failed: out of memory)' "$log"; then
            status=OOM
            [ "$expected" != oom ] || status=EXPECTED_OOM
        fi
    elif [ "$expected" = oom ]; then
        status=BASELINE_FITS
    fi
    printf '%s,%s,%s,%s\n' "$label" "$rc" "$status" "$log" >> "$RUN_DIR/status.csv"
    echo ">>> $label: $status (exit $rc)"
    case "$status" in
        PASS|EXPECTED_OOM|BASELINE_FITS) return 0 ;;
        *)
            failures=$((failures + 1))
            diagnostics > "$RUN_DIR/$label.diagnostics.log" 2>&1
            return 1 ;;
    esac
}

# Publish generated files only after success; failed generation cannot leave a
# partial trace/hot file that the next invocation mistakes for complete input.
ensure_generated() {
    local label="$1" target="$2"
    shift 2
    [ ! -s "$target" ] || return 0
    local tmp="$target.pending.$$.tmp"
    if run_logged "$label" normal "$@" --out "$tmp"; then
        mv "$tmp" "$target"
    else
        return 1
    fi
}

run_part() {
    local part="$1" model="$2"
    shift 2
    local res_csv="results/gpu/results_part_${part}.csv"
    local a c cfg tr_main tr_prof hot_bin
    local alphas caches
    if [ "$part" = b ]; then
        alphas=(0 0.6 1.0 1.2)
        caches=(0.0 0.01 0.05 0.1 0.25 0.5 1.0)
    else
        alphas=(0.6 1.0 1.2)
        caches=(0.05 0.1 0.25 0.5)
        if ensure_generated medium_trace data/medium.trace "$PY" tools/gen_trace.py configs/medium.cfg; then
            run_logged medium_baseline oom ./bin/gpu_single configs/medium.cfg "$res_csv" || true
        fi
    fi
    echo "Starting Part $part: $model; appending to $res_csv"
    for a in "${alphas[@]}"; do
        cfg="data/${model}_a${a}.cfg"
        tr_main="data/${model}_a${a}.trace"
        tr_prof="data/${model}_a${a}_prof.trace"
        hot_bin="data/${model}_a${a}_hot.bin"
        sed -E "s/^zipf_alpha[[:space:]]*=.*/zipf_alpha = $a/; s|^trace_file[[:space:]]*=.*|trace_file = $tr_main|" "configs/$model.cfg" > "$cfg"
        ensure_generated "${model}_${a}_main" "$tr_main" "$PY" tools/gen_trace.py "$cfg" || continue
        ensure_generated "${model}_${a}_profile" "$tr_prof" "$PY" tools/gen_trace.py "$cfg" --seed 1001 || continue
        ensure_generated "${model}_${a}_hot" "$hot_bin" "$PY" tools/profile_hot.py "$tr_prof" || continue
        for c in "${caches[@]}"; do
            run_logged "${model}_a${a}_cache${c}" normal ./bin/gpu_tier_pool "$cfg" "$c" "$hot_bin" "$res_csv" || true
        done
    done
    echo "Part $part sweep finished; see $RUN_DIR/status.csv for failures."
}

case "$MODE" in
    b|B) run_part b small ;;
    c|C) run_part c medium ;;
    all|ALL) run_part b small; run_part c medium ;;
    *) echo "Usage: $0 [b|c|all]"; exit 1 ;;
esac

echo "Finished: $failures unexpected failures; logs: $RUN_DIR"
# Report failure to automation only AFTER all configurations have been attempted.
[ "$failures" -eq 0 ]
