import os, sys, json
os.environ['POLARS_MAX_THREADS'] = '2'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
import numpy as np
import evalx
from tracking_cellmot.metrics import evaluate
K = evalx.K
S = np.array([1.625, 0.40625, 0.40625])
C = {r['movie']: r for r in json.load(open('/workspace/cl/p16/p19/v_dup/check_rows.json'))}


def rp(v):
    return np.array([max(0, int(round(float(v[k])))) for k in 'zyx']) * S


for mv in ['44b6_ddf577ad', '6bba_6ca87370', '6bba_f17befbc']:
    r = C[mv]
    nodes, edges = evalx.load_graph_json('/workspace/cl/p16/ps_p17_%s/graphs/%s.json' % (r['set'], mv))
    gt, nt = evalx.load_gt(mv)
    pred, mapping = evalx.to_graph(nodes, edges)
    er = evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    gE = set(zip(*[gt.edge_attrs()[c].to_list() for c in [K.EDGE_SOURCE, K.EDGE_TARGET]]))
    gE = {(int(a), int(b)) for a, b in gE}
    gnodes_on_track = {x for e in gE for x in e}
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    for kind in ['st', 'tt', 'par']:
        for n in r['rm_list'][kind]:
            inc = [(par[n], n)] if n in par else []
            inc += [(n, c) for c in out.get(n, [])]
            for a, b in inc:
                ga, gb = p2g.get(a), p2g.get(b)
                touches = (ga in gnodes_on_track) or (gb in gnodes_on_track)
                if not touches:
                    continue
                status = 'TP' if (ga, gb) in gE else 'FP'
                # neighbour context
                t = int(nodes[n]['t'])
                same = [(m, float(np.linalg.norm(rp(nodes[m]) - rp(nodes[n])))) for m in nodes if int(nodes[m]['t']) == t and m != n]
                same = sorted(same, key=lambda x: x[1])[:2]
                ctx = [(m, round(d, 2), 'par' if m in par else 'root', len(out.get(m, [])), 'gt=%s' % p2g.get(m)) for m, d in same]
                print(mv, kind, 'removed', n, 't', t, 'edge', (a, b), status, 'gt-match', (ga, gb), '| nearest same-frame', ctx)
