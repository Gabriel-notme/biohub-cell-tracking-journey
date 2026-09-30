#!/bin/bash
# P8 research run with candidate dumps; hold36 then prev4
cd /workspace/cl
for T in hold36 prev4; do
  PSTAGE_DUMP=/workspace/cl/cands_p8/$T SCRIPT=p_stage5.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p8_config.json $T p8
done
echo P8_RUNS_DONE
