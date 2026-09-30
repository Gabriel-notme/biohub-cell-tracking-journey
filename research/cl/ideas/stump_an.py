"""Read-only: stumps left behind by div_complete 'stolen' forks in P13. For each dc stolen fork (p->a,b; B5 parent q of b),
q's fate in P13 (present? children? history length, component size) and official matching of the stump nodes
(matched to GT, matched GT node out/in degree). Also the edge-level effect if the stump component were deleted."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def job(a):
    s, f = a
    name = Path(f).stem
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    bn, be = evalx.load_graph_json(B5[s] + '/' + name + '.json')
    bpar = {int(e['target_id']): int(e['source_id']) for e in be}
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(x)]: int(y) for x, y in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    gids = gt.node_ids(); gout = dict(zip(gids, gt.out_degree(gids))); gin = dict(zip(gids, gt.in_degree(gids)))
    ch = defaultdict(list); par = {}
    for e in edges: u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}; t = {n: int(v['t']) for n, v in nodes.items()}
    out = []
    for e in edges:
        if 'div_complete' not in e: continue
        p, b = int(e['source_id']), int(e['target_id'])
        q = bpar.get(b)
        if q is None or q == p: continue
        r = dict(set=s, movie=name, p=p, q=q, q_present=q in nodes)
        if q in nodes:
            # q's track (history) until root or fork
            h = [q]; x = q
            while x in par and len(ch[par[x]]) == 1: x = par[x]; h.append(x)
            root_is_fork = x in par
            r.update(q_kids=len(ch.get(q, [])), q_hist=len(h), q_root_fork=root_is_fork,
                     q_start_dp=None, q_matched=sum(n in p2g for n in h),
                     q_matched_annot=sum(1 for n in h if n in p2g and (gout[p2g[n]] > 0 or gin[p2g[n]] > 0)),
                     d_qp=float(np.linalg.norm(pos[q] - pos[p])))
            # distance of q's track start to p's lineage node at the same frame
            st = h[-1]; y = p
            while y in par and t[y] > t[st]: y = par[y]
            if t[y] == t[st]: r['q_start_dp'] = float(np.linalg.norm(pos[st] - pos[y]))
            # p-lineage node at q's start matched?
        out.append(r)
    return out


if __name__ == '__main__':
    jobs = [(s, f) for s in B5 for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(12) as p: R = [r for rs in p.map(job, jobs, chunksize=1) for r in rs]
    json.dump(R, open('/workspace/cl/ideas/stump_rows.json', 'w'))
    print('stolen dc forks', len(R), 'q present', sum(r['q_present'] for r in R))
    P = [r for r in R if r['q_present']]
    print('q kids', Counter(r['q_kids'] for r in P))
    E = [r for r in P if r['q_kids'] == 0]
    print('stump (q end) hist len', Counter(min(r['q_hist'], 12) for r in E))
    print('root is fork', Counter(r['q_root_fork'] for r in E))
    for L in [(1, 2), (3, 5), (6, 10), (11, 100)]:
        Q = [r for r in E if L[0] <= r['q_hist'] <= L[1]]
        print('hist %s n=%d  any node matched %d  any annotated-matched %d  start_dp<6um %d' % (L, len(Q), sum(r['q_matched'] > 0 for r in Q), sum(r['q_matched_annot'] > 0 for r in Q),
              sum(1 for r in Q if r['q_start_dp'] is not None and r['q_start_dp'] < 6)))
