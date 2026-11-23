#!/usr/bin/env bash
# Rebuilds everything the real-data traces need, doing only what is missing. Safe to rerun; quick when all is present.
# Called by verify.sh and experiments.sh (set PREPARE=0 to skip). Can also be run by hand: scripts/ghc/prepare_data.sh
#
# What it does:
#   1. venv + numpy/pyarrow (env.sh), and torch (CPU build) if the Meta converter needs it
#   2. Criteo: downloads one Hugging Face part file if missing, converts it to data/criteo.trace
#   3. Meta: converts $META_RAW to data/meta.trace. The raw file must be copied from your Mac (needs your password):
#        scp ~/dlrm-data/meta/fbgemm_t856_bs65536_0.pt.gz yosenl@ghc39.ghc.andrew.cmu.edu:/tmp/yosenl-data/meta/
#      If it is missing this script prints that command and moves on.
# Always exits 0; a missing dataset just means that tag is skipped later.
#
# Environment variables (optional): PY, META_RAW, CRITEO_DIR, CRITEO_PART

set -u
cd "$(dirname "$0")/../.."
. scripts/ghc/env.sh

META_RAW=${META_RAW:-/tmp/yosenl-data/meta/fbgemm_t856_bs65536_0.pt.gz}
CRITEO_DIR=${CRITEO_DIR:-/tmp/yosenl-data/criteo}
CRITEO_PART=${CRITEO_PART:-part-00079-99c339d5-fbac-4110-9dcf-75453a61a5c1.c000.snappy.parquet}
CRITEO_URL="https://huggingface.co/datasets/criteo/CriteoClickLogs/resolve/main/data/day=2015-02-15/$CRITEO_PART"

echo "== prepare data"

# Criteo
if [ -f data/criteo.trace ] && [ -f configs/criteo.cfg ]; then
    echo "  criteo: trace present"
else
    mkdir -p "$CRITEO_DIR"
    f=$CRITEO_DIR/$CRITEO_PART
    if [ ! -s "$f" ] || [ "$(wc -c < "$f")" -lt 1000000 ]; then
        echo "  criteo: downloading part file (~48 MB)"
        rm -f "$f"
        curl -fL -o "$f" "$CRITEO_URL" || { echo "  criteo: download failed, skipping"; rm -f "$f"; }
    fi
    if [ -s "$f" ]; then
        echo "  criteo: converting"
        "$PY" tools/convert_criteo.py "$f" --name criteo --batch-size 2048 --batches 60 > "$WORK/convert_criteo.log" 2>&1 \
            && echo "  criteo: done" || { echo "  criteo: conversion FAILED, see $WORK/convert_criteo.log"; tail -3 "$WORK/convert_criteo.log"; }
    fi
fi

# Meta
if [ -f data/meta.trace ] && [ -f configs/meta.cfg ]; then
    echo "  meta: trace present"
elif [ ! -s "$META_RAW" ]; then
    echo "  meta: raw file missing ($META_RAW), skipping. Run this on your Mac, then rerun:"
    echo "        ssh $(whoami)@$(hostname -f) 'mkdir -p $(dirname "$META_RAW")'   # once"
    echo "        scp ~/dlrm-data/meta/$(basename "$META_RAW") $(whoami)@$(hostname -f):$(dirname "$META_RAW")/"
else
    if ! "$PY" -c "import torch" > /dev/null 2>&1; then
        echo "  meta: installing torch (CPU build, a few hundred MB)"
        "$(dirname "$PY")/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu \
            || echo "  meta: torch install FAILED"
    fi
    echo "  meta: converting (loads a 2.4 GB file, may take a minute)"
    "$PY" tools/convert_meta.py "$META_RAW" --name meta --tables 8 --batch-size 2048 --batches 30 > "$WORK/convert_meta.log" 2>&1 \
        && echo "  meta: done" || { echo "  meta: conversion FAILED, see $WORK/convert_meta.log"; tail -3 "$WORK/convert_meta.log"; }
fi
exit 0
