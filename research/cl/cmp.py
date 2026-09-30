"""Compare configs pooled over hold36/audit32/prev4 rows with bootstrap. usage: cmp.py <rowsA_prefix> <rowsB_prefix>  (files <prefix><set>.json)"""
import sys, json, numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
A, B = sys.argv[1], sys.argv[2]
sets = sys.argv[3].split(',') if len(sys.argv) > 3 else ['hold36', 'audit32', 'prev4']
ra = {s: json.load(open(A + s + '.json')) for s in sets}; rb = {s: json.load(open(B + s + '.json')) for s in sets}
for s in sets:
    a, b = summarise(ra[s]), summarise(rb[s])
    print('%-8s A %.6f B %.6f  d %+.6f   divA %d/%d/%d divB %d/%d/%d  adjE A %.6f B %.6f' % (s, a['score'], b['score'], b['score'] - a['score'], a['division_tp'], a['division_fp'], a['division_fn'], b['division_tp'], b['division_fp'], b['division_fn'], a['adj_edge_jaccard'], b['adj_edge_jaccard']))
AA = [r for s in sets for r in ra[s]]; BB = [r for s in sets for r in rb[s]]
ka = {r['movie']: r for r in AA}; BB = [r for r in BB]; AA = [ka[r['movie']] for r in BB]
a, b = summarise(AA), summarise(BB)
print('POOLED A %.6f B %.6f d %+.6f  div A %d/%d/%d B %d/%d/%d' % (a['score'], b['score'], b['score'] - a['score'], a['division_tp'], a['division_fp'], a['division_fn'], b['division_tp'], b['division_fp'], b['division_fn']))
rng = np.random.default_rng(0); idx = np.arange(len(AA)); out = []
for _ in range(2000):
    k = rng.choice(idx, len(idx)); out.append(summarise([BB[i] for i in k])['score'] - summarise([AA[i] for i in k])['score'])
out = np.array(out); print('bootstrap mean %+.5f CI [%+.5f, %+.5f] P>0 %.3f' % (out.mean(), np.quantile(out, .025), np.quantile(out, .975), (out > 0).mean()))
