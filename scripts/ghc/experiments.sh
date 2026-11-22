#!/usr/bin/env bash
# CPU-reference baseline experiments (experiment set B): thread scaling, pooling sweep, skew sweep.
# Run from anywhere:  scripts/ghc/experiments.sh
# Build first (make). Results go to $WORK/results, one CSV per run, and a summary is printed at the end.
#
# Environment variables (all optional):
#   PY            python with numpy for gen_trace.py (default /tmp/yosenl-venv/bin/python)
#   WORK          scratch directory, keep it off AFS (default /tmp/yosenl-data)
#   TAGS          configs for thread scaling: names of configs/<tag>.cfg (default "small meta criteo");
#                 missing configs or traces are skipped with a note
#   THREADS       thread counts for scaling (default "1 2 4 8" plus all cores)
#   SWEEP         0 to skip the pooling and skew sweeps (default 1)
#   SWEEP_BASE    config the sweeps are derived from (default configs/small.cfg)
#   SWEEP_THREADS threads for the sweeps (default: all cores)
#   POOLING       pooling factors for the L sweep (default "1 4 16 64")
#   ALPHAS        Zipf alphas for the skew sweep (default "0 0.5 1.0")
#   SWEEP_BATCHES batches in sweep traces (default 20; the reference cycles through them)

set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/tmp/yosenl-venv/bin/python}
WORK=${WORK:-/tmp/yosenl-data}
NCPU=$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null) || echo 4)
THREADS=$(echo ${THREADS:-"1 2 4 8 $NCPU"} | tr ' ' '\n' | sort -un | tr '\n' ' ')
TAGS=${TAGS:-"small meta criteo"}
SWEEP=${SWEEP:-1}
SWEEP_BASE=${SWEEP_BASE:-configs/small.cfg}
SWEEP_THREADS=${SWEEP_THREADS:-$NCPU}
POOLING=${POOLING:-"1 4 16 64"}
ALPHAS=${ALPHAS:-"0 0.5 1.0"}
SWEEP_BATCHES=${SWEEP_BATCHES:-20}

RES=$WORK/results
CFG=$WORK/cfg
TR=$WORK/traces
mkdir -p "$RES" "$CFG" "$TR"

[ -x bin/cpu_ref ] || { echo "bin/cpu_ref not found: run make first"; exit 1; }

# run_one <tag> <config> <threads>: one cpu_ref run, results in $RES/<tag>__t<threads>.csv
run_one() {
    local tag=$1 cfg=$2 t=$3
    rm -f "$RES/${tag}__t$t.csv"
    printf "  %-14s threads=%-3s " "$tag" "$t"
    if OMP_NUM_THREADS=$t ./bin/cpu_ref "$cfg" "$RES/${tag}__t$t.csv" > "$RES/${tag}__t$t.log" 2>&1; then
        tail -1 "$RES/${tag}__t$t.log"
    else
        echo "FAILED (see $RES/${tag}__t$t.log)"
    fi
    rm -f ref_b*.bin   # dumps land in the current directory
}

# make_cfg <base.cfg> <out.cfg> <trace name> key=value...: copy of a config with some keys replaced
make_cfg() {
    local base=$1 out=$2 name=$3
    shift 3
    cp "$base" "$out"
    for kv in "$@"; do
        local k=${kv%%=*} v=${kv#*=}
        sed -i.bak "s|^$k[[:space:]]*=.*|$k = $v|" "$out"
    done
    sed -i.bak "s|^trace_file[[:space:]]*=.*|trace_file = $TR/$name.trace|" "$out"
    rm -f "$out.bak"
}

# ensure_trace <cfg>: generate the trace named in the config if it is missing
ensure_trace() {
    local tf
    tf=$(sed -n 's/^trace_file[[:space:]]*=[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$1")
    [ -f "$tf" ] || "$PY" tools/gen_trace.py "$1" > /dev/null 2>&1
    [ -f "$tf" ]
}

echo "== B1. thread scaling (threads: $THREADS)"
for tag in $TAGS; do
    cfg=configs/$tag.cfg
    if [ ! -f "$cfg" ]; then echo "  skip $tag: no $cfg"; continue; fi
    if ! ensure_trace "$cfg"; then echo "  skip $tag: trace missing and could not be generated"; continue; fi
    for t in $THREADS; do run_one "$tag" "$cfg" "$t"; done
done

if [ "$SWEEP" = 1 ]; then
    echo "== B2. pooling sweep (L = $POOLING), $SWEEP_THREADS threads, base $SWEEP_BASE"
    for L in $POOLING; do
        make_cfg "$SWEEP_BASE" "$CFG/pool$L.cfg" "pool$L" "pooling=$L" "num_batches=$SWEEP_BATCHES"
        if ensure_trace "$CFG/pool$L.cfg"; then run_one "pool$L" "$CFG/pool$L.cfg" "$SWEEP_THREADS"; else echo "  skip pool$L: trace generation failed"; fi
    done

    echo "== B3. skew sweep (alpha = $ALPHAS), $SWEEP_THREADS threads, base $SWEEP_BASE"
    for a in $ALPHAS; do
        make_cfg "$SWEEP_BASE" "$CFG/alpha$a.cfg" "alpha$a" "zipf_alpha=$a" "num_batches=$SWEEP_BATCHES"
        if ensure_trace "$CFG/alpha$a.cfg"; then run_one "alpha$a" "$CFG/alpha$a.cfg" "$SWEEP_THREADS"; else echo "  skip alpha$a: trace generation failed"; fi
    done
fi

echo
python3 scripts/ghc/summarize.py "$RES" | tee "$RES/summary.txt"
echo
echo "summary saved to $RES/summary.txt"
