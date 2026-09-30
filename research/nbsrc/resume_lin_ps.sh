#!/bin/bash
# resume_lin_ps.sh <V> <gpus> <gpu offset for P-stage>: lineage for <V>_h0 (skips done movies), expose <V>_h0_all, then P20 P-stage on the half
V=$1; G=$2; O=$3
(cd /workspace/nbsrc && python3 -u lin_run.py /workspace/nbrun/${V}_h0/working/reference_graphs /workspace/nbrun/${V}_h0/lin /workspace/data/train 10 $G > /workspace/nbrun/${V}_h0/lin2.log 2>&1 < /dev/null)
echo "LIN_END ${V}_h0 $(date -u +%H:%M) $(ls /workspace/nbrun/${V}_h0/lin | grep -c json) errors $(grep -c ' error ' /workspace/nbrun/${V}_h0/lin2.log)"
A=/workspace/nbrun/${V}_h0_all; mkdir -p $A; ln -sfn /workspace/nbrun/${V}_h0/lin $A/lineage_graphs; ln -sfn /workspace/nbrun/${V}_h0/working/reference_graphs $A/reference_graphs; ln -sfn /workspace/nbrun/${V}_h0/working/fullgraphs $A/fullgraphs
/workspace/nbsrc/run_ps_h0.sh ${V}_h0 p20_config.json $O > /workspace/nbrun/ps_${V}_h0_p20.out 2>&1 < /dev/null
echo "PS_END ${V}_h0 $(date -u +%H:%M)"
