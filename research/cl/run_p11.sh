#!/bin/bash
cd /workspace/cl
while tmux has-session -t p10 2>/dev/null; do sleep 20; done
for T in hold36 prev4 audit32 t127a t127b; do
  SCRIPT=p_stage4.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p11_config.json $T p11
done
echo P11_DONE
