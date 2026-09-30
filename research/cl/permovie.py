import json, sys
A = {r['movie']: r for r in json.load(open('/workspace/cl/rev/p3_hold36.json'))}
B = {r['movie']: r for r in json.load(open('/workspace/cl/rev/p5_hold36.json'))}
rows = []
for m in A:
    a, b = A[m], B[m]
    w = b['edge_tp'] + b['edge_fp'] + b['edge_fn']
    rows.append((b['adj_edge_jaccard'] - a['adj_edge_jaccard'], m, b['edge_tp'] - a['edge_tp'], b['edge_fp'] - a['edge_fp'], b['edge_fn'] - a['edge_fn'], a['num_pred_nodes'], w, (b['adj_edge_jaccard'] - a['adj_edge_jaccard']) * w))
rows.sort()
print('dadjE     movie            dTP dFP dFN  nodes   weight  weighted')
for r in rows: print('%+.5f %s %+4d %+4d %+4d %7d %6d %+8.3f' % r)
import numpy as np
dn = [r for r in rows if r[0] < 0]; up = [r for r in rows if r[0] > 0]
print('down: n %d sum weighted %.2f  dFP %d dTP %d | up: n %d sum weighted %.2f dFP %d dTP %d' % (len(dn), sum(r[7] for r in dn), sum(r[3] for r in dn), sum(r[2] for r in dn), len(up), sum(r[7] for r in up), sum(r[3] for r in up), sum(r[2] for r in up)))
