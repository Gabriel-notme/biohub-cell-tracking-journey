#!/bin/bash
cd /workspace/cl
while tmux has-session -t p12 2>/dev/null; do sleep 15; done
for T in t127a t127b; do
  SCRIPT=p_stage4.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p6_config.json $T p6n
done
echo P6TRAIN_DONE
