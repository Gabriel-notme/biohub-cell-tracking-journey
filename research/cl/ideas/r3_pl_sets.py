"""Per-set x embryo breakdown of a rule_eval JSON (official summarise), movie win/loss, edge TP/FP deltas.
usage: python3 ideas/r3_pl_sets.py <rule_eval json> [vi ...]"""
import sys, json
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
R = json.load(open(sys.argv[1]))
vis = [int(x) for x in sys.argv[2:]] or sorted({r['vi'] for r in R if r['vi'] > 0})
base = {r['movie']: r for r in R if r['vi'] == 0}
for vi in vis:
    cur = {r['movie']: r for r in R if r['vi'] == vi}
    print('== vi', vi)
    groups = [('all', lambda m: True), ('44b6', lambda m: m.startswith('44b6')), ('6bba', lambda m: m.startswith('6bba'))]
    for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']:
        for e in ['44b6', '6bba']:
            groups.append(('%s/%s' % (s, e), lambda m, s=s, e=e: base[m]['set'] == s and m.startswith(e)))
    groups.append(('b5clean(h36+p4+a32)', lambda m: base[m]['set'] in ('hold36', 'prev4', 'audit32')))
    groups.append(('b5train(t127)', lambda m: base[m]['set'] in ('t127a', 't127b')))
    for g, f in groups:
        mm = [m for m in base if f(m)]
        if not mm: continue
        a = summarise([base[m] for m in mm]); b = summarise([cur[m] for m in mm])
        dtp = sum(cur[m]['edge_tp'] - base[m]['edge_tp'] for m in mm); dfp = sum(cur[m]['edge_fp'] - base[m]['edge_fp'] for m in mm)
        up = sum(1 for m in mm if cur[m]['edge_tp'] - cur[m]['edge_fp'] > base[m]['edge_tp'] - base[m]['edge_fp'])
        dn = sum(1 for m in mm if cur[m]['edge_tp'] - cur[m]['edge_fp'] < base[m]['edge_tp'] - base[m]['edge_fp'])
        print('  %-22s n=%3d  d=%+.5f  eTP %+5d eFP %+5d  movies up %3d down %3d' % (g, len(mm), b['score'] - a['score'], dtp, dfp, up, dn))
