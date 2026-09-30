"""Strict acceptance check for a candidate rule (per-movie rows written by rule_eval.py: vi=0 base, vi=k candidate).
usage: strict.py <rows.json> <vi> [label]
S1 all-199 delta > 0 and bootstrap CI lower bound > 0
S2 each embryo delta > 0 and per-embryo bootstrap P(>0) >= 0.95
S3 clean40 (hold36+prev4) delta > 0
S4 B5-clean72 (hold36+prev4+audit32) delta >= 0
S5 leave-one-set-out: all 5 deltas > 0
S6 drop the 5 largest contributing movies overall: delta > 0; per embryo drop its 3 largest contributors: delta >= 0
S7 movies improved > movies worsened (by weighted adj-edge change)"""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')

rows, vi = json.load(open(sys.argv[1])), int(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else ''
base = {r['movie']: r for r in rows if r['vi'] == 0}; cur = {r['movie']: r for r in rows if r['vi'] == vi}
ms = sorted(base)
rng = np.random.default_rng(0)


def d(sel):
    return summarise([cur[m] for m in sel])['score'] - summarise([base[m] for m in sel])['score'] if sel else 0.0


def boot(sel, n=600):
    out = []
    for _ in range(n):
        k = rng.integers(0, len(sel), len(sel)); s = [sel[i] for i in k]
        out.append(summarise([cur[m] for m in s])['score'] - summarise([base[m] for m in s])['score'])
    return np.array(out)


def contrib(m):  # weighted adjusted-edge change of one movie (plus division part in raw counts)
    w = base[m]['edge_tp'] + base[m]['edge_fp'] + base[m]['edge_fn']
    return (cur[m]['adj_edge_jaccard'] - base[m]['adj_edge_jaccard']) * w + 0.5 * ((cur[m]['division_tp'] - base[m]['division_tp']) - 0.4 * (cur[m]['division_fp'] - base[m]['division_fp']))


res = {}
b = boot(ms); res['S1'] = (d(ms) > 0 and np.quantile(b, .025) > 0, 'all %+.5f CI [%+.5f, %+.5f]' % (d(ms), np.quantile(b, .025), np.quantile(b, .975)))
s2 = []; ok2 = True
for e in ['44b6', '6bba']:
    sel = [m for m in ms if m.startswith(e)]; be = boot(sel); p = (be > 0).mean()
    s2.append('%s %+.5f P>0 %.3f' % (e, d(sel), p)); ok2 &= d(sel) > 0 and p >= 0.95
res['S2'] = (ok2, ' | '.join(s2))
c40 = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]
res['S3'] = (d(c40) > 0, 'clean40 %+.5f (44b6 %+.5f, 6bba %+.5f)' % (d(c40), d([m for m in c40 if m.startswith('44b6')]), d([m for m in c40 if m.startswith('6bba')])))
c72 = [m for m in ms if base[m]['set'] in ('hold36', 'prev4', 'audit32')]
res['S4'] = (d(c72) >= 0, 'B5clean72 %+.5f (44b6 %+.5f, 6bba %+.5f)' % (d(c72), d([m for m in c72 if m.startswith('44b6')]), d([m for m in c72 if m.startswith('6bba')])))
loso = {s: d([m for m in ms if base[m]['set'] != s]) for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']}
res['S5'] = (all(v > 0 for v in loso.values()), ' '.join('-%s %+.5f' % kv for kv in loso.items()))
top = sorted(ms, key=contrib, reverse=True)
d5 = d([m for m in ms if m not in set(top[:5])])
per = {}
for e in ['44b6', '6bba']:
    sel = [m for m in ms if m.startswith(e)]; t3 = set(sorted(sel, key=contrib, reverse=True)[:3]); per[e] = d([m for m in sel if m not in t3])
res['S6'] = (d5 > 0 and all(v >= 0 for v in per.values()), 'drop top5 %+.5f | 44b6 drop top3 %+.5f | 6bba drop top3 %+.5f' % (d5, per['44b6'], per['6bba']))
up = sum(1 for m in ms if contrib(m) > 1e-9); dn = sum(1 for m in ms if contrib(m) < -1e-9)
res['S7'] = (up > dn, 'up %d down %d' % (up, dn))
print('== STRICT', label, 'vi=%d' % vi, 'PASS' if all(v[0] for v in res.values()) else 'FAIL', '(failed: %s)' % ','.join(k for k, v in res.items() if not v[0]))
for k, (ok, s) in res.items(): print('  %s %s  %s' % (k, 'ok ' if ok else 'XX ', s))
# informational S7b: only movies whose evaluable edge/division counts changed (pure node-count changes excluded)
KEYS = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn']
chg = [m for m in ms if any(cur[m][k] != base[m][k] for k in KEYS)]
up2 = sum(1 for m in chg if contrib(m) > 1e-9); dn2 = sum(1 for m in chg if contrib(m) < -1e-9)
ev_up = sum(1 for m in chg if (cur[m]['edge_tp'] - base[m]['edge_tp']) - 0.93 * (cur[m]['edge_fp'] - base[m]['edge_fp']) > 0)
ev_dn = sum(1 for m in chg if (cur[m]['edge_tp'] - base[m]['edge_tp']) - 0.93 * (cur[m]['edge_fp'] - base[m]['edge_fp']) < 0)
print('  S7b(info) movies with evaluable changes %d: score up %d down %d | edge-count net up %d down %d' % (len(chg), up2, dn2, ev_up, ev_dn))
