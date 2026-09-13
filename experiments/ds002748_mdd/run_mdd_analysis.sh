#!/usr/bin/env bash
set -o pipefail

cd "$(cd "$(dirname "$0")" && pwd)" || exit 1

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

run() {
    echo "[$1]"
    shift
    python3 "$@"
    local r=$?
    if [ $r -ne 0 ]; then
        echo "FAILED rc=$r: $*" >&2
        exit $r
    fi
}

run SEVERITY severity_regression_mdd.py
run ABLATION ablations_mdd.py

echo "MDD analysis complete"
exit 0
