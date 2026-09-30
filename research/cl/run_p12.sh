#!/bin/bash
cd /workspace/cl
for T in hold36 prev4 audit32; do
  SCRIPT=p_stage7.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p12_config.json $T p12
done
echo P12_DONE
