#!/bin/bash
cd /workspace/cl/p16/check2/reimpl
for K in '{"start_r": 2.0, "par_r": 3.0, "margin": 1.0, "bminlen": 5}' '{"start_r": 3.0, "par_r": 4.0, "margin": 3.0, "bminlen": 7}' '{"start_r": 3.5, "par_r": 5.0, "margin": 4.0, "bminlen": 8}'; do
  J=$(python3 -c "import json,sys; k=json.loads(sys.argv[1]); k['variants']=[{'dup': {'order': 'st', 'm_child': True, 'par': 'v2'}}]; print(json.dumps(k))" "$K")
  T=$(python3 -c "import json,sys; k=json.loads(sys.argv[1]); print('_'.join('%s%g' % (a, b) for a, b in k.items()))" "$K")
  RULE_POOL=6 python3 gcmp.py "$J" gcmp_grid_$T.json > gcmp_grid_$T.out 2>&1
done
echo GRID_FIN
