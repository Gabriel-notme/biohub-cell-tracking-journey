"""Score every existing fork F->(c1,c2) of the P15 graphs (all 5 sets) with the LOEO DivNet ensemble (44b6 movies with the models
trained on 6bba, and vice versa). usage: score_forks.py '<models glob with {E}>' <out.json>"""
import os, sys, json, glob
sys.path.insert(0, '/workspace/cl/p16/divnet')
from collections import defaultdict
import numpy as np
import dn_patch
os.environ['DN_MODELS'] = sys.argv[1]
res = {}
for st in ['hold36', 'prev4', 'audit32', 't127a', 't127b']:
    for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % st)):
        name = os.path.basename(f)[:-5]; d = json.load(open(f))
        nodes = {int(k): v for k, v in d['nodes'].items()}; out = defaultdict(list)
        for e in d['edges']: out[int(e['source_id'])].append(int(e['target_id']))
        tr = [(p, c[0], c[1]) for p, c in out.items() if len(c) == 2]
        if not tr: res[name] = {}; continue
        other = {'44b6': '6bba', '6bba': '44b6'}[name[:4]]
        s = dn_patch.divnet_scores('/workspace/data/train/%s.zarr' % name, nodes, tr, other)
        res[name] = {str(p): float(x) for (p, a, b), x in zip(tr, s)}
json.dump(res, open(sys.argv[2], 'w')); print('FORKS_SCORED', len(res), sum(len(v) for v in res.values()))
