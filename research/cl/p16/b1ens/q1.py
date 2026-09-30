"""Q1/Q2 end-to-end tables from cl/rev/<cfg>_<set>.json (official metric).
usage: q1.py <cfg,cfg,...>   (prints score per subset; singles = cfgs matching be_s*; ensemble = be_E5)"""
import sys, json
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')
import numpy as np
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
def load(cfg):
    R = {}
    for s in SETS:
        try:
            for r in json.load(open('/workspace/cl/rev/%s_%s.json' % (cfg, s))): R[r['movie']] = dict(r, set=s)
        except FileNotFoundError: pass
    return R
cfgs = sys.argv[1].split(','); D = {c: load(c) for c in cfgs}
ms = sorted(set.intersection(*[set(D[c]) for c in cfgs])); print('movies', len(ms))
SUB = {'all': lambda m, s: True, '44b6': lambda m, s: m.startswith('44b6'), '6bba': lambda m, s: m.startswith('6bba'),
       'clean40': lambda m, s: s in ('hold36', 'prev4'), 'clean40_44b6': lambda m, s: s in ('hold36', 'prev4') and m.startswith('44b6'),
       'clean40_6bba': lambda m, s: s in ('hold36', 'prev4') and m.startswith('6bba')}
res = {}
print('%-12s' % 'cfg' + ''.join('%-24s' % k for k in SUB))
for c in cfgs:
    line = '%-12s' % c; res[c] = {}
    for k, f in SUB.items():
        rr = [D[c][m] for m in ms if f(m, D[c][m]['set'])]
        if not rr: line += '%-24s' % '-'; continue
        sm = summarise(rr); res[c][k] = sm['score']
        line += '%.5f %3d/%3d       ' % (sm['score'], sm['division_tp'], sm['division_fp'])
    print(line)
sing = [c for c in cfgs if c.startswith('be_s')]
if len(sing) > 1:
    print('singles', sing)
    for k in SUB:
        if k not in res[sing[0]]: continue
        v = np.array([res[c][k] for c in sing]); line = '  %-13s singles min %.5f mean %.5f max %.5f sd %.5f' % (k, v.min(), v.mean(), v.max(), v.std(ddof=1))
        if 'be_E5' in res: line += ' | E5 %.5f (E5-mean %+.5f, E5-S0 %+.5f, rank %d/%d)' % (res['be_E5'][k], res['be_E5'][k] - v.mean(), res['be_E5'][k] - res['be_s0'][k], 1 + int((v > res['be_E5'][k]).sum()), len(v) + 1)
        print(line)
json.dump(res, open('/workspace/cl/p16/b1ens/q1_scores.json', 'w'), indent=1)
