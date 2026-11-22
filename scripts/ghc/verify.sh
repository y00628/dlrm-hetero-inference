#!/usr/bin/env bash
# Verification checks for the CPU reference and phase-0 tooling (experiment set A).
# Run from anywhere:  scripts/ghc/verify.sh
#
# Environment variables (all optional):
#   PY          python with numpy (default /tmp/yosenl-venv/bin/python), only used to generate traces
#   WORK        scratch directory, keep it off AFS (default /tmp/yosenl-data/verify)
#   THREADS     thread count for the "many threads" runs (default: all cores)
#   CXXFLAGS_OVERRIDE  replaces CXXFLAGS in the Makefile, e.g. "-O2 -std=c++17 -Wall -Iinclude" on a Mac without OpenMP
#   MAC_DUMPS   directory with ref_b{0,1}_{pooled,ctr}.bin made on another machine/build for `tiny`;
#               if set, they are compared with this machine's output (scalar build vs AVX2 build)
#   EXTRA_CFGS  config files to check with trace_info, default: configs/meta.cfg configs/criteo.cfg (if present)
#
# Exit code is 0 only if every check passed.

set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/tmp/yosenl-venv/bin/python}
WORK=${WORK:-/tmp/yosenl-data/verify}
NCPU=$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null) || echo 4)
THREADS=${THREADS:-$NCPU}
CXXFLAGS_OVERRIDE=${CXXFLAGS_OVERRIDE:-}
EXTRA_CFGS=${EXTRA_CFGS:-"configs/meta.cfg configs/criteo.cfg"}

mkdir -p "$WORK"
pass=0
fail=0
ok()  { echo "PASS: $1"; pass=$((pass + 1)); }
bad() { echo "FAIL: $1"; fail=$((fail + 1)); }

DUMPS="ref_b0_pooled ref_b0_ctr ref_b1_pooled ref_b1_ctr"

# run_ref <config> <threads> <outdir>: runs cpu_ref and moves the dumps into outdir
run_ref() {
    mkdir -p "$3"
    OMP_NUM_THREADS=$2 ./bin/cpu_ref "$1" "$WORK/scratch.csv" > "$3/log.txt" 2>&1 || return 1
    mv ref_b*.bin "$3/" 2>/dev/null
    [ -f "$3/ref_b0_pooled.bin" ]
}

# same_dumps <dirA> <dirB> <label>
same_dumps() {
    local allok=1
    for d in $DUMPS; do
        ./bin/check "$1/$d.bin" "$2/$d.bin" > "$WORK/check.out" 2>&1 || { allok=0; cat "$WORK/check.out"; }
    done
    [ $allok = 1 ] && ok "$3" || bad "$3"
}

echo "== 1. machine facts -> $WORK/machine.txt"
{
    hostname; date
    nvidia-smi 2>&1 | head -20
    nvidia-smi topo -m 2>&1 | head -20
    lscpu 2>&1 | head -25
    free -g 2>&1
    echo "ulimit -l: $(ulimit -l)"
    nvcc --version 2>&1 | tail -2
} > "$WORK/machine.txt" 2>&1
echo "saved"

echo "== 2. build"
make clean > /dev/null 2>&1
if make ${CXXFLAGS_OVERRIDE:+"CXXFLAGS=$CXXFLAGS_OVERRIDE"} > "$WORK/build.log" 2>&1; then ok "make"; else bad "make (see $WORK/build.log)"; fi
warns=$(grep -c "warning" "$WORK/build.log")
echo "   compiler warnings: $warns"
[ -x bin/cpu_ref ] && [ -x bin/check ] && [ -x bin/trace_info ] && ok "binaries exist" || bad "binaries missing"

echo "== 3. traces and trace_info"
for c in tiny small; do
    [ -f "data/$c.trace" ] || "$PY" tools/gen_trace.py "configs/$c.cfg" > /dev/null 2>&1
    if ./bin/trace_info "configs/$c.cfg" | tail -1 | grep -q '^OK$'; then ok "trace_info $c"; else bad "trace_info $c"; fi
done
for c in $EXTRA_CFGS; do
    [ -f "$c" ] || { echo "   skip $c (no config)"; continue; }
    if ./bin/trace_info "$c" | tail -1 | grep -q '^OK$'; then ok "trace_info $c"; else bad "trace_info $c"; fi
done

echo "== 4. cpu_ref on tiny: 1 thread vs $THREADS threads, and repeat run"
if run_ref configs/tiny.cfg 1 "$WORK/t1" && run_ref configs/tiny.cfg "$THREADS" "$WORK/tN" && run_ref configs/tiny.cfg "$THREADS" "$WORK/tN2"; then
    same_dumps "$WORK/t1" "$WORK/tN" "1 thread == $THREADS threads (tolerance 1e-4)"
    same_dumps "$WORK/tN" "$WORK/tN2" "repeat run with $THREADS threads"
    # a repeat run with the same thread count should be bit-identical
    cmp -s "$WORK/tN/ref_b0_ctr.bin" "$WORK/tN2/ref_b0_ctr.bin" && ok "repeat run is bit-identical" || echo "   note: repeat run not bit-identical (allowed, but unexpected with static scheduling)"
else
    bad "cpu_ref failed on tiny (see $WORK/t1/log.txt)"
fi

echo "== 5. checker catches errors"
if [ -f "$WORK/t1/ref_b0_ctr.bin" ]; then
    cp "$WORK/t1/ref_b0_ctr.bin" "$WORK/bad_nan.bin"
    printf '\x00\x00\xc0\x7f' | dd of="$WORK/bad_nan.bin" bs=1 seek=0 conv=notrunc 2> /dev/null   # first value -> NaN
    ./bin/check "$WORK/t1/ref_b0_ctr.bin" "$WORK/bad_nan.bin" > /dev/null 2>&1 && bad "checker accepted a NaN" || ok "checker rejects a NaN"
    head -c 100 "$WORK/t1/ref_b0_ctr.bin" > "$WORK/bad_short.bin"
    ./bin/check "$WORK/t1/ref_b0_ctr.bin" "$WORK/bad_short.bin" > /dev/null 2>&1 && bad "checker accepted a truncated file" || ok "checker rejects a truncated file"
    ./bin/check "$WORK/t1/ref_b0_ctr.bin" "$WORK/t1/ref_b0_ctr.bin" > /dev/null 2>&1 && ok "checker accepts identical files" || bad "checker rejected identical files"
fi

echo "== 6. other machine/build (MAC_DUMPS)"
if [ -n "${MAC_DUMPS:-}" ]; then
    if [ -f "$MAC_DUMPS/ref_b0_pooled.bin" ]; then
        same_dumps "$MAC_DUMPS" "$WORK/t1" "dumps from $MAC_DUMPS match this machine (tiny)"
    else
        bad "MAC_DUMPS=$MAC_DUMPS has no ref_b0_pooled.bin"
    fi
else
    echo "   skipped (set MAC_DUMPS=<dir with tiny dumps from another build> to compare)"
    echo "   tiny dumps from this run are in $WORK/t1 if you want to copy them elsewhere"
fi

echo
echo "RESULT: $pass passed, $fail failed"
[ $fail = 0 ]
