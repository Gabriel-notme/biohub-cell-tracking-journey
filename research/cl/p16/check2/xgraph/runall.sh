#!/bin/bash
cd /workspace/cl/p16/check2/xgraph
export RULE_POOL=6
for src in p15 b5 p13 p14 p17; do
  python xg.py $src > xg_$src.log 2>&1
done
cd /workspace/cl
for src in p15 b5 p13 p14 p17; do
  python rule_eval.py $src ideas.chk2_xgraph_$src '[{}]' > /workspace/cl/p16/check2/xgraph/re_$src.log 2>&1
  cp /workspace/cl/rule_eval_last_ideas_chk2_xgraph_$src.json /workspace/cl/p16/check2/xgraph/re_rows_$src.json
done
echo ALLDONE > /workspace/cl/p16/check2/xgraph/ALLDONE
