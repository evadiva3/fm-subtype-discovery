#!/usr/bin/env bash
set -o pipefail

cd "$(cd "$(dirname "$0")" && pwd)" || exit 1

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

n=${1:-100}

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

run SEARCH  hyperparameter_search_mdd.py "$n"
run TRAIN   train_mdd.py
run CLUSTER clustering_mdd.py

echo "MDD pipeline complete"
exit 0
