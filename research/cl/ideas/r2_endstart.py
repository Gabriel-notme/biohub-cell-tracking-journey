"""Read-only: free END(t) -> START(t+1) adjacencies in P14 (after trim,long) within 7 um, split by whether the END was CREATED by
term_trim (had a child in P13) or pre-existing, with GT label of the would-be edge (tp / evaluable fp / non-evaluable) per embryo.
Also: pre-ILP candidate status/prob of the would-be edge."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
for p in ['/workspace/cl', '/workspace/cl/ideas', '/workspace/official/src', '/workspace/code', '/workspace/p56stage']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def job(f):
    import evalx, combo14
    from edge_link import load_full
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = Path(f).stem; s = f.split('ps_p13_')[1].split('/')[0]
    n13, e13 = evalx.load_graph_json(f)
    had_child13 = {int(e['source_id']) for e in e13}
    fg = FULL[s] + '/' + name + '.geff'
    nodes, edges, _ = combo14.apply(n13, e13, order='trim,long', fullgeff=fg)
    fids, fT, fV, fE, fprob = load_full(fg)
    fp = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs(); GE = set(zip([int(a) for a in ea[K.EDGE_SOURCE].to_list()], [int(b) for b in ea[K.EDGE_TARGET].to_list()]))
    gout = Counter(a for a, b in GE); gin = Counter(b for a, b in GE)
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    rows = []
    for n in nodes:
        if out[n] or n not in par: continue
        t = int(nodes[n]['t'])
        for m in byt.get(t + 1, []):
            if m in par or not out[m]: continue
            d = float(np.linalg.norm(pos[n] - pos[m]))
            if d > 7: continue
            ga, gb = p2g.get(n), p2g.get(m)
            lab = 'tp' if (ga is not None and gb is not None and (ga, gb) in GE) else (
                'fpe' if ((ga is not None and gout[ga] > 0) or (gb is not None and gin[gb] > 0)) else 'ne')
            rows.append(dict(m=name, s=s, d=round(d, 2), created=int(n in had_child13), lab=lab, fprob=fp.get((n, m))))
    return rows


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(40, maxtasksperchild=4) as p: RR = p.map(job, fs, chunksize=1)
    rows = [r for rs in RR for r in rs]
    json.dump(rows, open('/workspace/cl/ideas/r2_endstart.json', 'w'))
    for cr in [1, 0]:
        for db in [(0, 3.5), (3.5, 5), (5, 7)]:
            for emb in ['44b6', '6bba']:
                sel = [r for r in rows if r['created'] == cr and db[0] < r['d'] <= db[1] and r['m'].startswith(emb)]
                print('created=%d d%s %s n=%4d' % (cr, db, emb, len(sel)), dict(Counter(r['lab'] for r in sel)),
                      'fullcand', sum(r['fprob'] is not None for r in sel))
