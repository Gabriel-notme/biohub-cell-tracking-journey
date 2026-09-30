"""Audit: relink accepts s->d and cuts s->cd and cs->d, then marks cs and cd 'touched'. For typ-3 relinks (a swap), was the
partner cs->cd re-linked later (by edge_link), and what does GT say about cs->cd? Diagnostic only (GT only for labels)."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def job(f):
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    name = Path(f).stem; s = f.split('ps_p13_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    b5 = json.load(open(B5[s] + '/' + name + '.json'))
    c5 = defaultdict(list); p5 = {}
    for e in b5['edges']:
        x, y = int(e['source_id']), int(e['target_id']); c5[x].append(y); p5[y] = x
    succ = defaultdict(list); par = {}; kind = {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); succ[x].append(y); par[y] = x
        kind[(x, y)] = 'relink' if 'relink' in e else ('edge_link' if 'edge_link' in e else ('div' if 'div_complete' in e else 'b5'))
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ea = gt.edge_attrs(); gE = set(zip(map(int, ea[K.EDGE_SOURCE].to_list()), map(int, ea[K.EDGE_TARGET].to_list())))
    gout = Counter(x for x, y in gE)

    def glab(x, y):
        a, b = p2g.get(x), p2g.get(y)
        if a is None and b is None: return 'unann'
        if a is not None and b is not None and (a, b) in gE: return 'GT'
        if (a is not None and gout.get(a)) : return 'FPeval'
        return 'unann'
    cat = Counter()
    for (x, y), k in kind.items():
        if k != 'relink': continue
        cd = [c for c in c5.get(x, []) if c != y]; cs = p5.get(y)
        if len(c5.get(x, [])) != 1 or cs is None or cs == x: continue
        cd = c5[x][0]
        if cd == y: continue
        # typ-3 swap
        st = 'partner_linked_' + kind[(cs, cd)] if (cs, cd) in kind else (
            'cs_has_child' if succ.get(cs) else ('cd_has_parent' if cd in par else ('cs_or_cd_pruned' if (cs not in nodes or cd not in nodes) else 'both_free')))
        cat[st + '|' + glab(cs, cd) + '|relink:' + glab(x, y)] += 1
    return cat


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/ps_p13_*/graphs/*.json'))
    with Pool(40) as p: R = p.map(job, files, chunksize=1)
    c = Counter()
    for r in R: c.update(r)
    for k, v in sorted(c.items()): print(k, v)
