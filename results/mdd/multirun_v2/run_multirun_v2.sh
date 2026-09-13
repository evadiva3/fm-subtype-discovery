#!/usr/bin/env bash
set -o pipefail
RP=/Users/evangeline.bangsil/localwork/UTD-PROJECT/fm-subtype-discovery
EX="$RP/experiments/ds002748_mdd"
OT="$RP/results/mdd"; MR="$OT/multirun_v2"; BR="$MR/by_run"
CN="$OT/canonical_single"
PY=/opt/anaconda3/envs/UTD/bin/python
NT=${1:-25}; N=${2:-20}
export OMP_NUM_THREADS=1
mkdir -p "$BR"; cd "$EX" || exit 1
LG="$MR/multirun_v2.log"
if [ ! -s "$CN/Embeddings.npy" ]; then
  echo "FATAL: $CN/Embeddings.npy missing" >&2
  exit 2
fi
CS=$(shasum -a 256 "$CN/Embeddings.npy" | cut -d' ' -f1)
echo "start $(date) NTRIALS=$NT N=$N" | tee "$LG"
echo "canonical sha256=$CS" | tee -a "$LG"
for r in $(seq 1 "$N"); do
  if [ -s "$BR/run${r}_params.json" ] && [ -s "$BR/run${r}_labels.csv" ]; then
    echo "RUN $r SKIP" | tee -a "$LG"; continue
  fi
  echo "RUN $r begin $(date)" | tee -a "$LG"; t0=$(date +%s)
  $PY hyperparameter_search_mdd.py "$NT" > "$BR/run${r}_search.log" 2>&1; rs=$?
  cp "$RP/data/tune/bestParamsMdd.json" "$BR/run${r}_params.json" 2>/dev/null
  t1=$(date +%s)
  $PY train_mdd.py > "$BR/run${r}_train.log" 2>&1; rt=$?; t2=$(date +%s)
  RD="$MR/_scratch/run${r}"; mkdir -p "$RD"
  MDD_OUT_DIR="$RD" $PY clustering_mdd.py > "$BR/run${r}_cluster.log" 2>&1; rc=$?; t3=$(date +%s)
  cp "$RD/Embeddings.npy"         "$BR/run${r}_Embeddings.npy"  2>/dev/null
  cp "$RD/K-Means-Labeling.csv"   "$BR/run${r}_labels.csv"      2>/dev/null
  cp "$RD/silhouette-scores.csv"  "$BR/run${r}_sil.csv"         2>/dev/null
  cp "$RD/bootstrap_summary.json" "$BR/run${r}_bootstrap.json"  2>/dev/null
  rm -rf "$RD"
  echo "RUN $r rc=$rs/$rt/$rc search=$((t1-t0))s train=$((t2-t1))s cluster=$((t3-t2))s total=$((t3-t0))s" | tee -a "$LG"
done
if [ "$(shasum -a 256 "$CN/Embeddings.npy" | cut -d' ' -f1)" != "$CS" ]; then
  echo "FATAL: canonical checkpoint changed during the protocol" | tee -a "$LG" >&2
  exit 3
fi
echo "canonical intact" | tee -a "$LG"
echo "done $(date)" | tee -a "$LG"
