# Shared setup for scripts/ghc/*.sh. Source it from the repo root; do not run it directly.
# Safe to run repeatedly. /tmp on GHC is per-host and gets cleaned, and the AFS home has a small quota, so this:
#   - recreates the python venv (numpy, pyarrow) at the directory containing $PY if it is missing
#   - creates the scratch directory /tmp/yosenl-data and $WORK
#   - keeps data/ (traces, dumps, results.csv) OFF AFS: data/ becomes a symlink to $DATA_DIR (default
#     /tmp/yosenl-data/traces). An existing real data/ directory is moved there, a dangling symlink is repaired.
#     Set DATA_LINK=0 to leave data/ alone.
# It also defines trace_ok <config>: true if the config's trace loads cleanly (trace_info prints OK), which is
# stricter than "the file exists" and catches truncated files left by a full disk.

if [ -d /usr/local/cuda/bin ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

PY=${PY:-/tmp/yosenl-venv/bin/python}
mkdir -p /tmp/yosenl-data "${WORK:-/tmp/yosenl-data}"

if [ ! -x "$PY" ]; then
    venv=$(dirname "$(dirname "$PY")")
    echo "setup: creating venv $venv (numpy, pyarrow)"
    if python3 -m venv "$venv" && "$venv/bin/pip" install -q numpy pyarrow; then
        echo "setup: venv ready"
    else
        echo "setup: WARNING could not create venv; trace generation will fail"
    fi
fi

if [ "${DATA_LINK:-1}" = 1 ]; then
    DATA_DIR=${DATA_DIR:-/tmp/yosenl-data/traces}
    mkdir -p "$DATA_DIR"
    if [ -L data ]; then
        [ -e data ] || { echo "setup: data symlink was dangling, creating $(readlink data)"; mkdir -p "$(readlink data)"; }
    elif [ -d data ]; then
        echo "setup: data/ is a real directory (AFS quota!), moving it to $DATA_DIR"
        # everything in data/ is regenerable (traces, dumps, csv): move each entry, or drop the AFS copy if /tmp has one
        for f in data/* data/.[!.]*; do
            [ -e "$f" ] || continue
            if [ -e "$DATA_DIR/$(basename "$f")" ]; then rm -rf "$f"; else mv "$f" "$DATA_DIR"/; fi
        done
        if rmdir data 2> /dev/null; then
            ln -s "$DATA_DIR" data
        else
            echo "setup: WARNING data/ could not be moved completely, it stays on AFS"
        fi
    elif [ ! -e data ]; then
        ln -s "$DATA_DIR" data
    fi
elif [ ! -e data ]; then
    mkdir -p data
fi

# trace_ok <config>: the trace named in the config exists and loads (needs bin/trace_info, falls back to file exists)
trace_ok() {
    local tf
    tf=$(sed -n 's/^trace_file[[:space:]]*=[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$1")
    [ -n "$tf" ] && [ -s "$tf" ] || return 1
    if [ -x bin/trace_info ]; then
        bin/trace_info "$1" 2> /dev/null | tail -1 | grep -q '^OK$'
    else
        return 0
    fi
}
