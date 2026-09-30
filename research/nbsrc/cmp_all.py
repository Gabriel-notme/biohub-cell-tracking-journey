"""cmp_all.py <candidate rows tag> <ref tag> [<ref tag> ...]: official-metric comparison of review rows (/workspace/cl/rev/<tag>_<set>.json, 199 movies)
candidate vs each reference: delta on all / 44b6 / 6bba / clean40 / B5-unseen72 / each set, movie-bootstrap 95% CI and P(>0) (1000 draws),
exact drop-top-k (k=1,3,5 largest positive movies removed), division TP/FP/FN deltas, edge TP/FP/FN deltas, node delta, movies better/worse."""
import json, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import numpy as np
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
def load(tag):
    R = {}
    for s in SETS:
        for r in json.load(open('/workspace/cl/rev/%s_%s.json' % (tag, s))): r = dict(r); r['set'] = s; R[r['movie']] = r
    return R
def sc(R, ms): return summarise([R[m] for m in ms])['score']
C = load(sys.argv[1]); ms = sorted(C)
print('candidate %s: all %.5f | 44b6 %.5f | 6bba %.5f | clean40 %.5f | unseen72 %.5f' % (sys.argv[1], sc(C, ms), sc(C, [m for m in ms if m.startswith('44b6')]), sc(C, [m for m in ms if m.startswith('6bba')]),
      sc(C, [m for m in ms if C[m]['set'] in ('hold36', 'prev4')]), sc(C, [m for m in ms if C[m]['set'] in ('hold36', 'prev4', 'audit32')])))
rng = np.random.default_rng(0)
for rt in sys.argv[2:]:
    B = load(rt); assert sorted(B) == ms
    G = {'all': ms, '44b6': [m for m in ms if m.startswith('44b6')], '6bba': [m for m in ms if m.startswith('6bba')], 'clean40': [m for m in ms if C[m]['set'] in ('hold36', 'prev4')],
         'unseen72': [m for m in ms if C[m]['set'] in ('hold36', 'prev4', 'audit32')]}
    G.update({s: [m for m in ms if C[m]['set'] == s] for s in SETS})
    d = {g: sc(C, v) - sc(B, v) for g, v in G.items()}
    bs = []
    for _ in range(1000):
        sel = [ms[i] for i in rng.integers(0, len(ms), len(ms))]; bs.append(sc(C, sel) - sc(B, sel))
    bs = np.array(bs)
    per = {m: sc(C, [m]) - sc(B, [m]) for m in ms}
    top = sorted(ms, key=lambda m: -per[m])
    dk = {k: sc(C, [m for m in ms if m not in top[:k]]) - sc(B, [m for m in ms if m not in top[:k]]) for k in (1, 3, 5)}
    sa, sb = summarise([C[m] for m in ms]), summarise([B[m] for m in ms])
    print('vs %-14s all %+.5f CI [%+.5f,%+.5f] P(>0) %.3f | 44b6 %+.5f 6bba %+.5f clean40 %+.5f unseen72 %+.5f | sets %s | drop-top1/3/5 %+.5f/%+.5f/%+.5f | div TP %+d FP %+d FN %+d | edge TP %+d FP %+d FN %+d | nodes %+d | better/worse %d/%d' % (
        rt, d['all'], np.quantile(bs, .025), np.quantile(bs, .975), (bs > 0).mean(), d['44b6'], d['6bba'], d['clean40'], d['unseen72'], ' '.join('%s%+.5f' % (s[:3] + s[-1], d[s]) for s in SETS),
        dk[1], dk[3], dk[5], sa['division_tp'] - sb['division_tp'], sa['division_fp'] - sb['division_fp'], sa['division_fn'] - sb['division_fn'],
        sum(C[m]['edge_tp'] - B[m]['edge_tp'] for m in ms), sum(C[m]['edge_fp'] - B[m]['edge_fp'] for m in ms), sum(C[m]['edge_fn'] - B[m]['edge_fn'] for m in ms),
        sum(C[m]['num_pred_nodes'] - B[m]['num_pred_nodes'] for m in ms), sum(1 for m in ms if per[m] > 1e-9), sum(1 for m in ms if per[m] < -1e-9)))
