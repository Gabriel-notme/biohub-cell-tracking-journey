import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']: os.environ[k] = '1'
import sys, json
sys.path.insert(0, '/workspace/code')
import numpy as np
import evalx
from evalx import K
name = '44b6_7a302da0'
n15, e15 = evalx.load_graph_json('/workspace/cl/ps_p15_hold36/graphs/%s.json' % name)
n19, e19 = evalx.load_graph_json('/workspace/cl/p16/ps_p19r_hold36/graphs/%s.json' % name)
r15 = evalx.score_movie(name, n15, e15); r19 = evalx.score_movie(name, n19, e19)
keys = [k for k in r15 if isinstance(r15[k], (int, float)) and r15[k] != r19.get(k)]
print({k: (r15[k], r19[k]) for k in keys})
# GT divisions near parent 24134
gt, _ = evalx.load_gt(name)
na = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
ids = na[K.NODE_ID].to_list(); ts = na['t'].to_list(); P = np.stack([na[c].to_numpy() for c in 'zyx'], 1) * np.array(evalx.SCALE)
ea = gt.edge_attrs(); src = ea[K.EDGE_SOURCE].to_list(); dst = ea[K.EDGE_TARGET].to_list()
from collections import Counter
oc = Counter(src)
idx = {i: j for j, i in enumerate(ids)}
p = n15[24134]; pp = np.array([p['z'], p['y'], p['x']], float) * np.array(evalx.SCALE)
for g, c in oc.items():
    if c >= 2:
        j = idx[g]
        if abs(ts[j] - int(p['t'])) <= 3:
            d = float(np.linalg.norm(P[j] - pp))
            if d < 15: print('GT fork', g, 't', ts[j], 'dist to pred parent %.2f' % d)
