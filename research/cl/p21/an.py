"""Compare variants in a rule_eval rows file: an.py <rows.json> <base_vi> <vi,vi,...>  (official metric; movie bootstrap CI; drop-top-k)."""
import sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import numpy as np
R = json.load(open(sys.argv[1])); b = int(sys.argv[2]); cands = [int(x) for x in sys.argv[3].split(',')]
by = {}
for x in R: by.setdefault(x['vi'], {})[x['movie']] = x
base = by[b]; ms = sorted(base)
G = {'all': lambda m: True, '44b6': lambda m: m.startswith('44b6'), '6bba': lambda m: m.startswith('6bba'),
     'clean40': lambda m: base[m]['set'] in ('hold36', 'prev4'), 'unseen72': lambda m: base[m]['set'] in ('hold36', 'prev4', 'audit32')}
rng = np.random.default_rng(0)
def sc(D, sel): return summarise([D[m] for m in sel])['score']
for vi in cands:
    cur = by[vi]; out = []
    for g, f in G.items():
        sel = [m for m in ms if f(m)]; out.append('%s %+.5f' % (g, sc(cur, sel) - sc(base, sel)))
    A = summarise([cur[m] for m in ms]); B = summarise([base[m] for m in ms])
    per = ' '.join('%s%+.5f' % (s[:3] + s[-1], sc(cur, [m for m in ms if base[m]['set'] == s]) - sc(base, [m for m in ms if base[m]['set'] == s])) for s in ('hold36', 'prev4', 'audit32', 't127a', 't127b'))
    bs = []
    for _ in range(400):
        sel = [ms[i] for i in rng.integers(0, len(ms), len(ms))]; bs.append(sc(cur, sel) - sc(base, sel))
    bs = np.array(bs)
    dm = {m: sc(cur, [m]) - sc(base, [m]) for m in ms}; top = sorted(ms, key=lambda m: -dm[m])
    dk = [sc(cur, [m for m in ms if m not in top[:k]]) - sc(base, [m for m in ms if m not in top[:k]]) for k in (1, 3, 5)]
    print('vi%d vs vi%d: %s | CI [%+.5f,%+.5f] P>0 %.2f | %s | drop1/3/5 %+.5f/%+.5f/%+.5f | divTP %+d FP %+d | edgeTP %+d FP %+d FN %+d | nodes %+d | better/worse %d/%d' % (
        vi, b, ' '.join(out), np.quantile(bs, .025), np.quantile(bs, .975), (bs > 0).mean(), per, dk[0], dk[1], dk[2], A['division_tp'] - B['division_tp'], A['division_fp'] - B['division_fp'],
        sum(cur[m]['edge_tp'] - base[m]['edge_tp'] for m in ms), sum(cur[m]['edge_fp'] - base[m]['edge_fp'] for m in ms), sum(cur[m]['edge_fn'] - base[m]['edge_fn'] for m in ms),
        sum(cur[m]['num_pred_nodes'] - base[m]['num_pred_nodes'] for m in ms), sum(1 for d in dm.values() if d > 1e-9), sum(1 for d in dm.values() if d < -1e-9)))
