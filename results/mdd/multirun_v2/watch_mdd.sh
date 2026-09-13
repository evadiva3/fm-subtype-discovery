#!/usr/bin/env bash
MR=/Users/evangeline.bangsil/localwork/UTD-PROJECT/fm-subtype-discovery/results/mdd/multirun_v2
while true; do
  clear
  echo "refreshed $(date '+%H:%M:%S')"
  if pgrep -f run_multirun_v2 >/dev/null; then
    echo "status running"
  else
    echo "status stopped"
  fi
  dn=$(grep -c '^RUN .* rc=' "$MR/multirun_v2.log" 2>/dev/null)
  echo "runs ${dn:-0}/20"
  grep '^RUN .* rc=' "$MR/multirun_v2.log" 2>/dev/null | tail -8
  cu=$(ls -t "$MR"/by_run/*_search.log 2>/dev/null | head -1)
  if [ -n "$cu" ]; then
    tl=$(grep -cE 'Trial [0-9]+ (finished|pruned)' "$cu" 2>/dev/null)
    echo "search $(basename "$cu") trials ${tl:-0}/25"
    tail -3 "$cu"
  fi
  if [ "${dn:-0}" -gt 0 ]; then
    sc=$(grep -oE 'total=[0-9]+' "$MR/multirun_v2.log" | cut -d= -f2 | awk '{s+=$1} END {print int(s/NR)}')
    lf=$(( (20 - dn) * sc ))
    echo "eta $((sc/60))min/run ~$((lf/3600))h $(((lf%3600)/60))m"
  fi
  sleep 30
done
