#!/bin/bash
cd /workspace/cl
for T in hold36 prev4 audit32 t127a t127b; do
  SCRIPT=p_stage4.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p13ns_config.json $T p13ns
done
echo P13NS_DONE
