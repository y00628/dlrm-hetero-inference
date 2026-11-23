# Shared setup for scripts/ghc/*.sh. Source it from the repo root; do not run it directly.
# Safe to run repeatedly. /tmp on GHC is per-host and gets cleaned, so this rebuilds what is missing:
#   - the python venv (numpy, pyarrow) at the directory containing $PY
#   - the scratch directory /tmp/yosenl-data and $WORK
#   - the data/ directory, or the target of a dangling data -> /tmp/... symlink

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

if [ -L data ] && [ ! -e data ]; then
    echo "setup: data symlink was dangling, creating $(readlink data)"
    mkdir -p "$(readlink data)"
elif [ ! -e data ]; then
    mkdir -p data
fi
