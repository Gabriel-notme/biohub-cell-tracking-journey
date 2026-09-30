#!/bin/bash
# chain_h0.sh <V> <gpus>: wait for <V>_h0 base run, lineage (8 workers), then expose /workspace/nbrun/<V>_h0_all for run_ps_h0.sh
V=$1; G=$2
busy() { ls -l /proc/[0-9]*/cwd 2>/dev/null | grep -q "/workspace/nbrun/$1\(/\|$\)"; }
sleep 60; while busy ${V}_h0; do sleep 30; done; echo "BASE_END ${V}_h0 $(date -u +%H:%M) refs $(ls /workspace/nbrun/${V}_h0/working/reference_graphs | wc -l)"
(cd /workspace/nbsrc && python3 -u lin_run.py /workspace/nbrun/${V}_h0/working/reference_graphs /workspace/nbrun/${V}_h0/lin /workspace/data/train 10 $G > /workspace/nbrun/${V}_h0/lin.log 2>&1 < /dev/null)
echo "LIN_END ${V}_h0 $(date -u +%H:%M) $(ls /workspace/nbrun/${V}_h0/lin | grep -c json) errors $(grep -c ' error ' /workspace/nbrun/${V}_h0/lin.log)"
A=/workspace/nbrun/${V}_h0_all; mkdir -p $A; ln -sfn /workspace/nbrun/${V}_h0/lin $A/lineage_graphs; ln -sfn /workspace/nbrun/${V}_h0/working/reference_graphs $A/reference_graphs; ln -sfn /workspace/nbrun/${V}_h0/working/fullgraphs $A/fullgraphs
echo "CHAIN_DONE ${V}_h0 $(date -u +%H:%M)"
