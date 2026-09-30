"""Verifier: per-movie concentration of r3_rlf deltas (official summarise). usage: python3 ideas/r3_rlf_vchk.py <rows.json>"""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
R = json.load(open(sys.argv[1]))
base = {r['movie']: r for r in R if r['vi'] == 0}
ms = sorted(base)
for vi in sorted({r['vi'] for r in R if r['vi'] > 0}):
    cur = {r['movie']: r for r in R if r['vi'] == vi}
    print('===== vi', vi)
    for emb in ['44b6', '6bba', '']:
        mm = [m for m in ms if m.startswith(emb)]
        d = summarise([cur[m] for m in mm])['score'] - summarise([base[m] for m in mm])['score']
        net = {m: (cur[m]['edge_tp'] - cur[m]['edge_fp']) - (base[m]['edge_tp'] - base[m]['edge_fp']) for m in mm}
        dtp = {m: cur[m]['edge_tp'] - base[m]['edge_tp'] for m in mm}
        dfp = {m: cur[m]['edge_fp'] - base[m]['edge_fp'] for m in mm}
        chg = [m for m in mm if dtp[m] or dfp[m]]
        up = [m for m in mm if net[m] > 0]; dn = [m for m in mm if net[m] < 0]
        # jackknife: embryo delta with each movie dropped
        jk = {}
        for m in chg:
            rest = [x for x in mm if x != m]
            jk[m] = summarise([cur[x] for x in rest])['score'] - summarise([base[x] for x in rest])['score']
        order = sorted(mm, key=lambda m: -net[m])
        def drop(k):
            rest = [x for x in mm if x not in order[:k]]
            return summarise([cur[x] for x in rest])['score'] - summarise([base[x] for x in rest])['score']
        tot = sum(net.values()); pos = sum(v for v in net.values() if v > 0)
        print('  %-4s n=%3d d=%+.5f changed=%3d up=%3d down=%3d sumTP %+d sumFP %+d net %+d | drop1 %+.5f drop2 %+.5f drop3 %+.5f | jk_min %+.5f (%s)' % (
            emb or 'all', len(mm), d, len(chg), len(up), len(dn), sum(dtp.values()), sum(dfp.values()), tot, drop(1), drop(2), drop(3),
            min(jk.values()) if jk else 0, min(jk, key=jk.get) if jk else '-'))
        top = order[:4] + [m for m in order[::-1][:3]]
        print('     top/bottom:', ', '.join('%s[%s] %+d' % (m, base[m]['set'], net[m]) for m in top))
    # sign test over movies with net != 0
    from math import comb
    netall = [(cur[m]['edge_tp'] - cur[m]['edge_fp']) - (base[m]['edge_tp'] - base[m]['edge_fp']) for m in ms]
    u = sum(v > 0 for v in netall); dd = sum(v < 0 for v in netall); n = u + dd
    p = sum(comb(n, k) for k in range(u, n + 1)) / 2 ** n if n else 1
    print('  sign test all: up %d down %d one-sided p=%.4f' % (u, dd, p))
    for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']:
        for emb in ['44b6', '6bba']:
            mm = [m for m in ms if base[m]['set'] == s and m.startswith(emb)]
            if not mm: continue
            d = summarise([cur[m] for m in mm])['score'] - summarise([base[m] for m in mm])['score']
            net = sum((cur[m]['edge_tp'] - cur[m]['edge_fp']) - (base[m]['edge_tp'] - base[m]['edge_fp']) for m in mm)
            print('     %-8s %s n=%3d d=%+.5f net %+d' % (s, emb, len(mm), d, net))
