#!/bin/bash
cd /workspace/cl
for T in hold36 prev4 audit32 t127a t127b; do
  SCRIPT=p_stage6.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p10_config.json $T p10
done
echo P10_DONE
