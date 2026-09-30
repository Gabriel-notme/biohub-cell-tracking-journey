"""Build extended-relink candidate rows on P13 graphs (all 199 movies) with GT labels:
pos = (s,d) is a GT edge; new_ev = new edge evaluable; old_tp = removed edge is a GT edge; old_ev = removed edge evaluable.
gain if applied = pos - old_tp  (TP)  and  (new_ev & !pos) - (old_ev & !old_tp)  (FP). -> /workspace/cl/xrl/<set>__<movie>.pkl"""
import os, sys, json, glob, pickle
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
SRC = os.environ.get('XRL_SRC', 'p13')
OUT = Path('/workspace/cl/xrl_%s' % SRC); OUT.mkdir(exist_ok=True)


def job(args):
    s, f = args
    name = Path(f).stem; of = OUT / ('%s__%s.pkl' % (s, name))
    if of.exists(): return 1
    import evalx, xrl, edge_link
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    fp = Path(FULL[s]) / (name + '.geff')
    full = edge_link.load_full(fp) if fp.exists() else None
    rows = xrl.features(nodes, edges, full)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    ma = _matched_node_attrs(_match_full(pred, gt, evalx.SCALE, 7.))
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ea = gt.edge_attrs(); GE = set(); gs = defaultdict(list); gp = defaultdict(list)
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): GE.add((int(x), int(y))); gs[int(x)].append(int(y)); gp[int(y)].append(int(x))
    isge = lambda a, b: p2g.get(a) is not None and p2g.get(b) is not None and (p2g[a], p2g[b]) in GE
    ev = lambda a, b: (p2g.get(a) is not None and bool(gs.get(p2g[a]))) or (p2g.get(b) is not None and bool(gp.get(p2g[b])))
    for r in rows:
        old = (r['o'], r['d']) if r['typ'] == 0 else (r['s'], r['o'])
        r['pos'] = int(isge(r['s'], r['d'])); r['new_ev'] = int(ev(r['s'], r['d']))
        r['old_tp'] = int(isge(*old)); r['old_ev'] = int(ev(*old))
    pickle.dump(rows, open(of, 'wb'))
    return 1


if __name__ == '__main__':
    G = {s: '/workspace/cl/ps_%s_%s/graphs' % (SRC, s) for s in FULL}
    jobs = [(s, f) for s in G for f in sorted(glob.glob(G[s] + '/*.json'))]
    with Pool(100) as p: print('done', sum(p.map(job, jobs, chunksize=1)), len(jobs))
