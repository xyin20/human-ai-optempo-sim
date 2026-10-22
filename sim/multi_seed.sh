#!/bin/bash
# Runs four more simulated sessions (seeds 8-11), then re-runs seed 7 so it stays the showcase session.
set -e
cd "$(dirname "$0")"
for s in 8 9 10 11; do
  python3 cosim.py $s > out/cosim_seed$s.log 2>&1
  python3 results.py > out/results_seed$s.log 2>&1
  cp out/results.json out/results_seed$s.json
  echo "seed $s done"
done
python3 cosim.py 7 > out/cosim_seed7.log 2>&1
python3 results.py > out/results_seed7.log 2>&1
cp out/results.json out/results_seed7.json
cp out/chat_team.json out/chat_team_seed7.json
python3 summarize.py
