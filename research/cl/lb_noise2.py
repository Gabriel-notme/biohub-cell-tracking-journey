"""How noisy is a leaderboard computed on k movies? Resample k movies from the 72 labelled validation movies and compute the
score difference between two configs (official aggregation) -> SD and P(sign flip)."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
def rows(cfg):
    out = {}
    for s in ['hold36', 'prev4', 'audit32']:
        for r in json.load(open('/workspace/cl/rev/%s_%s.json' % (cfg, s))): out[r['movie']] = r
    return out
rng = np.random.default_rng(0)
for a, b in [('p6n', 'p8'), ('p6n', 'p11'), ('p8', 'p11')]:
    A, B = rows(a), rows(b); ms = sorted(set(A) & set(B))
    full = summarise([B[m] for m in ms])['score'] - summarise([A[m] for m in ms])['score']
    line = '%s->%s  all72 d %+.4f |' % (a, b, full)
    for k in [58, 141]:
        d = []
        for _ in range(2000):
            idx = rng.choice(len(ms), k, replace=True); sub = [ms[i] for i in idx]
            d.append(summarise([B[m] for m in sub])['score'] - summarise([A[m] for m in sub])['score'])
        d = np.array(d); line += ' k=%d: mean %+.4f sd %.4f P(d<0) %.2f |' % (k, d.mean(), d.std(), (d < 0).mean())
    print(line)
# absolute score noise for one config
A = rows('p11'); ms = sorted(A)
for k in [58, 141]:
    v = [summarise([A[ms[i]] for i in rng.choice(len(ms), k, replace=True)])['score'] for _ in range(2000)]
    print('P11 absolute score on k=%d movies: sd %.4f, 5-95%% [%.4f, %.4f]' % (k, np.std(v), np.percentile(v, 5), np.percentile(v, 95)))
