#!/bin/bash
cd /workspace/cl
for T in hold36 prev4 audit32 t127a t127b; do
  SCRIPT=p_stage4.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p13_config.json $T p13
done
echo P13_DONE
