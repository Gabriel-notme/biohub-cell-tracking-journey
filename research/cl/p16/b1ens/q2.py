"""Q2 data scaling (pre-registered): full_k = be_s{k} vs half_k = be_h{k} on 44b6 movies, k = 0,1,2; paired movie bootstrap."""
import sys, json
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')
import numpy as np
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
def load(cfg):
    R = {}
    for s in SETS:
        for r in json.load(open('/workspace/cl/rev/%s_%s.json' % (cfg, s))):
            if r['movie'].startswith('44b6'): R[r['movie']] = dict(r, set=s)
    return R
F = [load('be_s%d' % k) for k in range(3)]; H = [load('be_h%d' % k) for k in range(3)]
ms = sorted(set.intersection(*[set(x) for x in F + H])); print('44b6 movies', len(ms))
sc = lambda R, sel: summarise([R[m] for m in sel])['score']
d = []
for k in range(3):
    sf, sh = summarise([F[k][m] for m in ms]), summarise([H[k][m] for m in ms])
    d.append(sf['score'] - sh['score'])
    print('seed %d full %.5f (div %d/%d) half %.5f (div %d/%d) delta %+.5f' % (k, sf['score'], sf['division_tp'], sf['division_fp'], sh['score'], sh['division_tp'], sh['division_fp'], d[-1]))
rng = np.random.default_rng(0); B = []
for _ in range(1000):
    s = [ms[i] for i in rng.integers(0, len(ms), len(ms))]
    B.append(np.mean([sc(F[k], s) - sc(H[k], s) for k in range(3)]))
B = np.array(B)
for lab, f in [('clean40', lambda m: F[0][m]['set'] in ('hold36', 'prev4')), ('t127+audit', lambda m: F[0][m]['set'] not in ('hold36', 'prev4'))]:
    sel = [m for m in ms if f(m)]; print('  %s n=%d mean delta %+.5f' % (lab, len(sel), np.mean([sc(F[k], sel) - sc(H[k], sel) for k in range(3)])))
print('mean delta %+.5f | seeds positive %d/3 | bootstrap P(>0) %.3f CI [%+.5f, %+.5f]' % (np.mean(d), sum(x > 0 for x in d), (B > 0).mean(), np.quantile(B, .025), np.quantile(B, .975)))
