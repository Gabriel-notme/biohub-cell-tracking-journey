#!/bin/bash
cd /workspace/cl
for T in hold36 prev4 audit32; do
  SCRIPT=p_stage6.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p9_config.json $T p9
done
echo P9_DONE
