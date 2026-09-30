#!/bin/bash
# chain2.sh <V> <gpus>: for each half already running as <V>_h0/<V>_h1: wait for its base run to end, run lineage (8 workers), then merge.
V=$1; G=$2
busy() { ls -l /proc/[0-9]*/cwd 2>/dev/null | grep -q "/workspace/nbrun/$1\(/\|$\)"; }
half() { sleep 60; while busy $1; do sleep 30; done; echo "BASE_END $1 $(date -u +%H:%M) refs $(ls /workspace/nbrun/$1/working/reference_graphs | wc -l)"
  (cd /workspace/nbsrc && python3 -u lin_run.py /workspace/nbrun/$1/working/reference_graphs /workspace/nbrun/$1/lin /workspace/data/train 8 $G > /workspace/nbrun/$1/lin.log 2>&1 < /dev/null)
  echo "LIN_END $1 $(date -u +%H:%M) $(ls /workspace/nbrun/$1/lin | grep -c json) errors $(grep -c ' error ' /workspace/nbrun/$1/lin.log)"; }
half ${V}_h0 & half ${V}_h1 & wait
A=/workspace/nbrun/${V}_all; mkdir -p $A/lineage_graphs $A/reference_graphs $A/fullgraphs
for h in h0 h1; do
  for f in /workspace/nbrun/${V}_$h/lin/*.json; do ln -sfn $f $A/lineage_graphs/; done
  for f in /workspace/nbrun/${V}_$h/working/reference_graphs/*.json; do ln -sfn $f $A/reference_graphs/; done
  for f in /workspace/nbrun/${V}_$h/working/fullgraphs/*.geff; do ln -sfn $f $A/fullgraphs/; done
done
echo "CHAIN_DONE $V $(date -u +%H:%M) lineage $(ls $A/lineage_graphs | wc -l) ref $(ls $A/reference_graphs | wc -l) full $(ls $A/fullgraphs | wc -l)"
