#!/usr/bin/env bash
# Checks and measurements for the single-GPU + CPU-tier phase. Run on GHC from anywhere:
#   scripts/ghc/phase1_eval.sh
#
# Part 1, correctness (gate): for each config in CORR_CFGS, compare the pooled embedding output of gpu_single,
#   gpu_tier_pool and gpu_tier_zerocopy with the CPU reference (bin/check, tolerance 1e-4), at several cache
#   sizes, and check that the two tiered paths agree with each other.
# Part 2, measurements: PCIe link info, bench_pcie, hot-row hit rates, and a cache-size sweep that runs
#   gpu_tier_pool and gpu_tier_zerocopy at the same hot list. A comparison table (pool vs zero-copy, PCIe MB with
#   and without CPU pooling) is written to $OUT/phase1_summary.md and copied to results/phase1/.
#
# Environment variables (all optional):
#   WORK         scratch directory, off AFS (default /tmp/yosenl-data)
#   OUT          output directory (default $WORK/phase1)
#   REPO_OUT     where the summary and small text outputs are copied for committing (default results/phase1)
#   CORR_CFGS    configs for part 1 (default "tiny small meta"); configs without a trace are skipped. tiny, small
#                and medium traces are generated if missing; meta and criteo come from prepare_data.sh
#   CORR_FRACS   cache fractions for part 1 (default "0 0.25 1.0": all cold, mixed, all hot)
#   SWEEP_CFGS   configs for part 2 (default "small meta")
#   SWEEP_FRACS  cache fractions for part 2 (default "0 0.01 0.05 0.1 0.25 0.5 1.0")
#   PART         "1", "2" or "all" (default all)
#   PREPARE      0 to skip prepare_data.sh
#   PY           python used to generate traces (default /tmp/yosenl-venv/bin/python)
#
# Exit code is 0 only if every correctness check passed. Build first happens inside (make).

set -u
cd "$(dirname "$0")/../.."

WORK=${WORK:-/tmp/yosenl-data}
. scripts/ghc/env.sh
OUT=${OUT:-$WORK/phase1}
REPO_OUT=${REPO_OUT:-results/phase1}
CORR_CFGS=${CORR_CFGS:-"tiny small meta"}
CORR_FRACS=${CORR_FRACS:-"0 0.25 1.0"}
SWEEP_CFGS=${SWEEP_CFGS:-"small meta"}
SWEEP_FRACS=${SWEEP_FRACS:-"0 0.01 0.05 0.1 0.25 0.5 1.0"}
PART=${PART:-all}
mkdir -p "$OUT"

pass=0
fail=0
ok()  { echo "PASS: $1"; pass=$((pass + 1)); echo "PASS: $1" >> "$OUT/correctness.txt"; }
bad() { echo "FAIL: $1"; fail=$((fail + 1)); echo "FAIL: $1" >> "$OUT/correctness.txt"; }
: > "$OUT/correctness.txt"

trace_of() { sed -n 's/^trace_file[[:space:]]*=[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$1"; }

# have_trace <tag>: true if configs/<tag>.cfg exists and its trace loads. A missing or truncated synthetic trace
# (tiny, small, medium) is regenerated; meta and criteo are never overwritten with synthetic data.
have_trace() {
    local cfg=configs/$1.cfg tf
    [ -f "$cfg" ] || { echo "  skip $1: no $cfg"; return 1; }
    tf=$(trace_of "$cfg")
    if ! trace_ok "$cfg"; then
        case $1 in
            tiny|small|medium)
                echo "  $1: trace missing or broken, regenerating"
                rm -f "$tf"
                "$PY" tools/gen_trace.py "$cfg" > "$OUT/gen_$1.log" 2>&1 || tail -3 "$OUT/gen_$1.log" | sed 's/^/      /'
                ;;
        esac
    fi
    trace_ok "$cfg" || { echo "  skip $1: trace $tf missing or does not load (disk quota? try: df -h $(dirname "$tf"))"; return 1; }
}

# try <logfile> <command...>: run, send output to the log, and on failure show the last lines of it
try() {
    local log=$1
    shift
    if "$@" > "$log" 2>&1; then return 0; fi
    echo "FAILED (exit $?, $log)"
    tail -4 "$log" | sed 's/^/      /'
    return 1
}

# chk <label> <refdump> <otherdump>
chk() {
    if ./bin/check "$2" "$3" > "$OUT/check.out" 2>&1; then ok "$1"; else bad "$1"; tail -2 "$OUT/check.out" | sed 's/^/      /'; fi
}

# take_dumps <cwd prefix> <dest prefix>: move <prefix>_b{0,1}_pooled.bin out of the current directory
take_dumps() { mv "$1_b0_pooled.bin" "$2_b0.bin" 2> /dev/null; mv "$1_b1_pooled.bin" "$2_b1.bin" 2> /dev/null; }

# make_hot <tag> <hotfile> <fractions as a space list>
make_hot() {
    local frac_csv
    frac_csv=$(echo "$3" | tr ' ' ',')
    ./bin/hot_profile "configs/$1.cfg" "$2" --fractions "$frac_csv" --pad 1.0 > "$OUT/hot_profile_$1.txt" 2>&1
}

echo "== build"
if make > "$OUT/build.log" 2>&1; then echo "  make ok"; else echo "  make FAILED, see $OUT/build.log"; tail -15 "$OUT/build.log"; exit 1; fi
for b in cpu_ref check hot_profile gpu_single gpu_tier_pool gpu_tier_zerocopy bench_pcie; do
    [ -x bin/$b ] || { echo "  missing bin/$b (is nvcc on the PATH?)"; exit 1; }
done
case "$CORR_CFGS $SWEEP_CFGS" in *meta*|*criteo*) [ "${PREPARE:-1}" = 1 ] && scripts/ghc/prepare_data.sh ;; esac

NCPU=$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null) || echo 4)

# ---------------------------------------------------------------- part 1
if [ "$PART" = all ] || [ "$PART" = 1 ]; then
    echo
    echo "== part 1: correctness vs the CPU reference (embedding stage, pooled dumps)"
    for tag in $CORR_CFGS; do
        have_trace "$tag" || continue
        cfg=configs/$tag.cfg
        d=$OUT/corr/$tag
        rm -rf "$d"; mkdir -p "$d"
        echo "-- $tag"
        rm -f ref_b*.bin
        if ! OMP_NUM_THREADS=$NCPU ./bin/cpu_ref "$cfg" "$d/ref.csv" > "$d/ref.log" 2>&1; then
            bad "$tag: cpu_ref failed (see $d/ref.log)"; tail -4 "$d/ref.log" | sed 's/^/      /'; continue
        fi
        take_dumps ref "$d/ref"
        rm -f ref_b*_ctr.bin   # only the pooled dumps are compared in this phase
        [ -f "$d/ref_b0.bin" ] || { bad "$tag: no reference dumps"; continue; }
        make_hot "$tag" "$d/hot" "$CORR_FRACS"

        rm -f gpu_b*_pooled.bin
        if ./bin/gpu_single "$cfg" "$d/gs.csv" > "$d/gs.log" 2>&1; then
            take_dumps gpu "$d/gs"
            chk "$tag: gpu_single == reference (batch 0)" "$d/ref_b0.bin" "$d/gs_b0.bin"
            chk "$tag: gpu_single == reference (batch 1)" "$d/ref_b1.bin" "$d/gs_b1.bin"
        else
            bad "$tag: gpu_single failed (see $d/gs.log)"; tail -4 "$d/gs.log" | sed 's/^/      /'
        fi

        for f in $CORR_FRACS; do
            rm -f tier_b*_pooled.bin zc_b*_pooled.bin
            if ./bin/gpu_tier_pool "$cfg" "$f" "$d/hot" "$d/tp.csv" > "$d/tp_$f.log" 2>&1; then
                take_dumps tier "$d/tp_$f"
                chk "$tag: gpu_tier_pool cache=$f == reference (batch 0)" "$d/ref_b0.bin" "$d/tp_${f}_b0.bin"
                chk "$tag: gpu_tier_pool cache=$f == reference (batch 1)" "$d/ref_b1.bin" "$d/tp_${f}_b1.bin"
            else
                bad "$tag: gpu_tier_pool cache=$f failed (see $d/tp_$f.log)"; tail -4 "$d/tp_$f.log" | sed 's/^/      /'
            fi
            if ./bin/gpu_tier_zerocopy "$cfg" "$f" "$d/hot" "$d/zc.csv" > "$d/zc_$f.log" 2>&1; then
                take_dumps zc "$d/zc_$f"
                chk "$tag: gpu_tier_zerocopy cache=$f == reference (batch 0)" "$d/ref_b0.bin" "$d/zc_${f}_b0.bin"
                chk "$tag: gpu_tier_zerocopy cache=$f == reference (batch 1)" "$d/ref_b1.bin" "$d/zc_${f}_b1.bin"
                [ -f "$d/tp_${f}_b0.bin" ] && chk "$tag: pool == zerocopy at cache=$f (batch 0)" "$d/tp_${f}_b0.bin" "$d/zc_${f}_b0.bin"
            else
                bad "$tag: gpu_tier_zerocopy cache=$f failed (see $d/zc_$f.log)"; tail -4 "$d/zc_$f.log" | sed 's/^/      /'
            fi
        done
    done
    rm -f ref_b*.bin gpu_b*_pooled.bin tier_b*_pooled.bin zc_b*_pooled.bin
    echo
    echo "correctness: $pass passed, $fail failed"
fi

# ---------------------------------------------------------------- part 2
if [ "$PART" = all ] || [ "$PART" = 2 ]; then
    echo
    echo "== part 2: measurements"
    # The idle link drops to Gen1 to save power, so sample the link generation while bench_pcie is running.
    echo "-- bench_pcie (256 MB) and PCIe link"
    ./bin/bench_pcie 256 > "$OUT/bench_pcie.txt" 2>&1 &
    bpid=$!
    maxgen=0; width=0
    while kill -0 "$bpid" 2> /dev/null; do
        read -r g w < <(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2> /dev/null | tr -d ',')
        [ "${g:-0}" -gt "$maxgen" ] 2> /dev/null && maxgen=$g
        width=${w:-$width}
        sleep 0.2
    done
    wait "$bpid"
    nvidia-smi --query-gpu=name,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader > "$OUT/pcie_link.txt" 2>&1
    echo "link gen seen during bench_pcie: $maxgen (x$width); max supported: $(cat "$OUT/pcie_link.txt")" | tee -a "$OUT/pcie_link.txt"
    grep '^|' "$OUT/bench_pcie.txt" || echo "  bench_pcie failed, see $OUT/bench_pcie.txt"

    for tag in $SWEEP_CFGS; do
        have_trace "$tag" || continue
        cfg=configs/$tag.cfg
        d=$OUT/sweep/$tag
        rm -rf "$d"; mkdir -p "$d"
        echo "-- $tag: hot-row profile"
        make_hot "$tag" "$d/hot" "$SWEEP_FRACS"
        sed -n '/cache/,$p' "$OUT/hot_profile_$tag.txt" | head -12
        try "$d/gs.log" ./bin/gpu_single "$cfg" "$d/results.csv" || echo "  (gpu_single failed for $tag)"
        for f in $SWEEP_FRACS; do
            printf "   cache=%-5s pool: " "$f"
            if try "$d/pool_$f.log" ./bin/gpu_tier_pool "$cfg" "$f" "$d/hot" "$d/results.csv"; then
                printf "%s  " "$(grep -o 'hit rate [0-9.]*' "$d/pool_$f.log")"
            fi
            printf "  zerocopy: "
            try "$d/zc_$f.log" ./bin/gpu_tier_zerocopy "$cfg" "$f" "$d/hot" "$d/results.csv" && echo ok
        done
        rm -f tier_b*_pooled.bin zc_b*_pooled.bin gpu_b*_pooled.bin
    done

    echo
    python3 scripts/ghc/phase1_summary.py "$OUT" "$SWEEP_FRACS" | tee "$OUT/phase1_summary.md"
    mkdir -p "$REPO_OUT"
    cp "$OUT/phase1_summary.md" "$OUT/pcie_link.txt" "$OUT/bench_pcie.txt" "$OUT/correctness.txt" "$REPO_OUT/" 2> /dev/null
    for tag in $SWEEP_CFGS; do cp "$OUT/hot_profile_$tag.txt" "$REPO_OUT/" 2> /dev/null; done
    echo
    echo "copied to $REPO_OUT/ (in the repo; commit it yourself). Per-run CSVs and logs stay in $OUT."
fi

echo
echo "RESULT: correctness $pass passed, $fail failed"
[ $fail = 0 ]
