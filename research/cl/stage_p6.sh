#!/bin/bash
set -e
python3 - <<'EOF'
import json
p4 = json.load(open('/workspace/p12ds/p4_config.json'))
c6 = dict(p4, variant='P6', relink={'th': 0.65, 'model': 'relink_lgb.json'}, edge_link={'th': 0.4, 'gap2': True, 'model': 'edge_lgb.json'})
json.dump(c6, open('/workspace/p56stage/p6_config.json', 'w'), indent=1)
print(json.dumps(c6))
EOF
cd /workspace/cl
bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p6_config.json prev4 p6
python /workspace/code/score_csv.py /workspace/cl/ps_p6_prev4/submission.csv 2>&1 | grep SUMMARY | cut -c1-400
