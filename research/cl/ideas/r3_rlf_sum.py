"""Subset breakdown with movie-bootstrap CIs (official summarise) for a rule_eval rows JSON.
usage: python3 ideas/r3_rlf_sum.py <json> [vi ...]"""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
R = json.load(open(sys.argv[1]))
vis = [int(x) for x in sys.argv[2:]] or sorted({r['vi'] for r in R if r['vi'] > 0})
base = {r['movie']: r for r in R if r['vi'] == 0}
CLEAN = ('hold36', 'prev4', 'audit32')
groups = [('all', lambda m: True), ('44b6', lambda m: m.startswith('44b6')), ('6bba', lambda m: m.startswith('6bba')),
          ('clean40', lambda m: base[m]['set'] in ('hold36', 'prev4')),
          ('clean40/44b6', lambda m: base[m]['set'] in ('hold36', 'prev4') and m.startswith('44b6')),
          ('clean40/6bba', lambda m: base[m]['set'] in ('hold36', 'prev4') and m.startswith('6bba')),
          ('b5clean72', lambda m: base[m]['set'] in CLEAN),
          ('b5clean72/44b6', lambda m: base[m]['set'] in CLEAN and m.startswith('44b6')),
          ('b5clean72/6bba', lambda m: base[m]['set'] in CLEAN and m.startswith('6bba')),
          ('audit32/44b6', lambda m: base[m]['set'] == 'audit32' and m.startswith('44b6')),
          ('t127', lambda m: base[m]['set'] in ('t127a', 't127b')),
          ('t127/44b6', lambda m: base[m]['set'] in ('t127a', 't127b') and m.startswith('44b6')),
          ('t127/6bba', lambda m: base[m]['set'] in ('t127a', 't127b') and m.startswith('6bba'))]
rng = np.random.default_rng(1)
for vi in vis:
    cur = {r['movie']: r for r in R if r['vi'] == vi}
    st = {k: sum(r.get(k, 0) for r in cur.values()) for k in cur[next(iter(cur))] if k.startswith('rlf_')}
    print('== vi', vi, json.dumps(st))
    for g, f in groups:
        mm = sorted(m for m in base if f(m))
        if not mm: continue
        d = summarise([cur[m] for m in mm])['score'] - summarise([base[m] for m in mm])['score']
        bs = []
        for _ in range(1000):
            k = rng.integers(0, len(mm), len(mm))
            bs.append(summarise([cur[mm[j]] for j in k])['score'] - summarise([base[mm[j]] for j in k])['score'])
        dtp = sum(cur[m]['edge_tp'] - base[m]['edge_tp'] for m in mm); dfp = sum(cur[m]['edge_fp'] - base[m]['edge_fp'] for m in mm)
        net = {m: (cur[m]['edge_tp'] - cur[m]['edge_fp']) - (base[m]['edge_tp'] - base[m]['edge_fp']) for m in mm}
        up = sum(v > 0 for v in net.values()); dn = sum(v < 0 for v in net.values())
        pos = sorted((v for v in net.values() if v > 0), reverse=True); tot = sum(net.values())
        top3 = sum(pos[:3]) / tot if tot > 0 else float('nan')
        print('  %-16s n=%3d d=%+.5f CI [%+.5f, %+.5f] eTP %+4d eFP %+4d up %3d down %3d top3/net %.2f' % (
            g, len(mm), d, np.quantile(bs, .025), np.quantile(bs, .975), dtp, dfp, up, dn, top3))
