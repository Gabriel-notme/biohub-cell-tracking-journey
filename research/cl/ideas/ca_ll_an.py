"""Diagnostic for ca_longlink: label each added long link (GT TP / evaluable FP / unannotated), with distance, p, fe, and
whether the GT source of a TP is a division parent. GT used only for labels."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
RMAX = float(os.environ.get('LL_RMAX', '20'))


def job(f):
    import evalx
    import ideas.ca_longlink as m
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    name = Path(f).stem; s = f.split('ps_p13_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    nn, ne, st = m.apply(nodes, edges, rmax=RMAX, fullgeff=FULL[s] + '/' + name + '.geff')
    added = [e for e in ne if 'long_link' in e]
    if not added: return []
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nn, ne); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ea = gt.edge_attrs(); gE = set(zip(map(int, ea[K.EDGE_SOURCE].to_list()), map(int, ea[K.EDGE_TARGET].to_list())))
    gout = Counter(x for x, y in gE); gin = Counter(y for x, y in gE)
    out = []
    for e in added:
        x, y = int(e['source_id']), int(e['target_id'])
        a, b = p2g.get(x), p2g.get(y)
        if a is not None and b is not None and (a, b) in gE: lab = 'TP'
        elif (a is not None and gout.get(a)) or (b is not None and gin.get(b)): lab = 'FPeval'
        else: lab = 'unann'
        d = float(np.linalg.norm((np.array([nodes[y][k] for k in 'zyx']) - np.array([nodes[x][k] for k in 'zyx'])) * S))
        out.append(dict(movie=name, set=s, lab=lab, dist=d, p=e['long_link'], t=int(nodes[x]['t']), src_div=bool(a is not None and gout.get(a, 0) >= 2)))
    return out


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/ps_p13_*/graphs/*.json'))
    with Pool(32, maxtasksperchild=1) as p: R = [x for r in p.map(job, files, chunksize=1) for x in r]
    for emb in ['44b6', '6bba']:
        rr = [r for r in R if r['movie'].startswith(emb)]
        print(emb, 'added', len(rr), Counter(r['lab'] for r in rr), 'src_div TP', sum(r['src_div'] and r['lab'] == 'TP' for r in rr),
              'movies with links', len({r['movie'] for r in rr}))
        for lo, hi in [(14, 16), (16, 18), (18, 20), (20, 30)]:
            q = [r for r in rr if lo < r['dist'] <= hi]
            print('   dist %d-%d' % (lo, hi), len(q), dict(Counter(r['lab'] for r in q)))
    c = Counter((r['set'], r['lab']) for r in R); print(sorted(c.items()))
    top = Counter(r['movie'] for r in R).most_common(8); print(top)
    json.dump(R, open('/workspace/cl/ideas/ca_ll_an.json', 'w'))
