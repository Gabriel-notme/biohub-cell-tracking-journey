"""Pair per-movie official-metric rows of two full runs (cl/rev/<cfg>_<set>.json) into strict.py format (vi=0 base, vi=1 cand)
and print a per-set / per-embryo breakdown with edge and division count changes.
usage: mk_rows.py <base cfg> <cand cfg> <sets comma> <out.json>"""
import sys, json
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')
A, B, sets, out = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4]
rows = []
for s in sets:
    a = {r['movie']: r for r in json.load(open('/workspace/cl/rev/%s_%s.json' % (A, s)))}
    b = {r['movie']: r for r in json.load(open('/workspace/cl/rev/%s_%s.json' % (B, s)))}
    miss = sorted(set(a) ^ set(b))
    if miss: print('WARNING movie mismatch in', s, miss[:5])
    for m in sorted(set(a) & set(b)):
        rows.append(dict(a[m], set=s, vi=0)); rows.append(dict(b[m], set=s, vi=1))
json.dump(rows, open(out, 'w'))


def show(label, sel):
    ra = [r for r in rows if r['vi'] == 0 and sel(r)]; rb = [r for r in rows if r['vi'] == 1 and sel(r)]
    if not ra: return
    sa, sb = summarise(ra), summarise(rb)
    k = lambda rr, f: sum(r[f] for r in rr)
    up = sum(1 for x, y in zip(ra, rb) if y['adj_edge_jaccard'] > x['adj_edge_jaccard'] + 1e-12 or y['division_tp'] - y['division_fp'] > x['division_tp'] - x['division_fp'])
    dn = sum(1 for x, y in zip(ra, rb) if y['adj_edge_jaccard'] < x['adj_edge_jaccard'] - 1e-12 or y['division_tp'] - y['division_fp'] < x['division_tp'] - x['division_fp'])
    print('%-10s n=%3d  %s %.5f -> %.5f  d=%+.5f | dTP %+d dFP %+d dFN %+d | div TP %+d FP %+d | nodes %+d | up %d down %d' % (
        label, len(ra), A, sa['score'], sb['score'], sb['score'] - sa['score'], k(rb, 'edge_tp') - k(ra, 'edge_tp'), k(rb, 'edge_fp') - k(ra, 'edge_fp'),
        k(rb, 'edge_fn') - k(ra, 'edge_fn'), k(rb, 'division_tp') - k(ra, 'division_tp'), k(rb, 'division_fp') - k(ra, 'division_fp'),
        k(rb, 'num_pred_nodes') - k(ra, 'num_pred_nodes'), up, dn))


show('all', lambda r: True)
for s in sets: show(s, lambda r, s=s: r['set'] == s)
for e in ['44b6', '6bba']: show(e, lambda r, e=e: r['movie'].startswith(e))
for s in sets:
    for e in ['44b6', '6bba']: show(s + ':' + e, lambda r, s=s, e=e: r['set'] == s and r['movie'].startswith(e))
