#!/bin/bash
cd /workspace/cl
[ -f /workspace/cl/ps_p3_prev4/submission.csv ] || bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p3_config.json prev4 p3 > /dev/null 2>&1
python train_xset.py h hold36
python train_xset.py p prev4
SRC=/workspace/cl/ps_p3_{set}/graphs
ONLY_SETS=hold36,prev4 python pipe_eval.py xbase $SRC relink:0.65 el:0.4 2>&1 | grep -v WARN | tail -n 1
ONLY_SETS=prev4 EL_MODEL=/workspace/cl/edge_lgb_h.json RL_MODEL=/workspace/cl/relink_lgb_h.json python pipe_eval.py xh $SRC relink:0.65 el:0.4 2>&1 | grep -v WARN | tail -n 1
ONLY_SETS=hold36 EL_MODEL=/workspace/cl/edge_lgb_p.json RL_MODEL=/workspace/cl/relink_lgb_p.json python pipe_eval.py xp $SRC relink:0.65 el:0.4 2>&1 | grep -v WARN | tail -n 1
python - <<'EOF'
import json, sys
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
def S(f): r = json.load(open(f)); s = summarise(r); return s['score'], s['adj_edge_jaccard'], sum(x['edge_tp'] for x in r), sum(x['edge_fp'] for x in r), sum(x['edge_fn'] for x in r)
print('prev4  base %.6f adjE %.6f TP/FP/FN %d/%d/%d' % S('/workspace/cl/rows/xbase_prev4.json'))
print('prev4  +hold36-trained %.6f adjE %.6f TP/FP/FN %d/%d/%d' % S('/workspace/cl/rows/xh_prev4.json'))
print('hold36 base %.6f adjE %.6f TP/FP/FN %d/%d/%d' % S('/workspace/cl/rows/xbase_hold36.json'))
print('hold36 +prev4-trained %.6f adjE %.6f TP/FP/FN %d/%d/%d' % S('/workspace/cl/rows/xp_hold36.json'))
EOF
