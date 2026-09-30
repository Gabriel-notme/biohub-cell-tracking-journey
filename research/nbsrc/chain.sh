#!/bin/bash
# chain.sh <variant tag prefix> <gpus pair> [KEY=VALUE overrides...]: base tracker h0 (must already be running as <V>_h0) -> base h1 on the same pair,
# lineage for each half as soon as its base run ends (8 workers on the same pair), then merge both halves into /workspace/nbrun/<V>_all.
V=$1; G=$2; shift 2
busy() { ls -l /proc/[0-9]*/cwd 2>/dev/null | grep -q "/workspace/nbrun/$1\(/\|$\)"; }
waitrun() { sleep 60; while busy $1; do sleep 30; done; echo "BASE_END $1 $(date -u +%H:%M) refs $(ls /workspace/nbrun/$1/working/reference_graphs | wc -l)"; }
lin() { (cd /workspace/nbsrc && OMP_NUM_THREADS=2 python3 -u lin_run.py /workspace/nbrun/$1/working/reference_graphs /workspace/nbrun/$1/lin /workspace/data/train 8 $G > /workspace/nbrun/$1/lin.log 2>&1 < /dev/null; echo "LIN_END $1 $(date -u +%H:%M) $(ls /workspace/nbrun/$1/lin | grep -c json)") ; }
waitrun ${V}_h0
if [ ! -f /workspace/nbrun/${V}_h1/run.py ]; then (cd /workspace/nbsrc && python3 gen_run2.py ${V}_h1 $G /workspace/nbdata/h1 "$@"); fi
(cd /workspace/nbrun/${V}_h1 && nohup python3 -u run.py > run.log 2>&1 < /dev/null &)
lin ${V}_h0 &
waitrun ${V}_h1
lin ${V}_h1 &
wait
A=/workspace/nbrun/${V}_all; mkdir -p $A/lineage_graphs $A/reference_graphs $A/fullgraphs
for h in h0 h1; do
  for f in /workspace/nbrun/${V}_$h/lin/*.json; do ln -sfn $f $A/lineage_graphs/; done
  for f in /workspace/nbrun/${V}_$h/working/reference_graphs/*.json; do ln -sfn $f $A/reference_graphs/; done
  for f in /workspace/nbrun/${V}_$h/working/fullgraphs/*.geff; do ln -sfn $f $A/fullgraphs/; done
done
echo "CHAIN_DONE $V $(date -u +%H:%M) lineage $(ls $A/lineage_graphs | wc -l) ref $(ls $A/reference_graphs | wc -l) full $(ls $A/fullgraphs | wc -l)"
