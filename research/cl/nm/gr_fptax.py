"""Taxonomy of P15 evaluable-FP edges (the 'cut' oracle mass) and of radius-oracle TP (FN edges between P15 nodes with no candidate).
FP edge a->b (a matched to GT ga with GT successor gs, or b matched to GT gb with GT predecessor):
  flip      : gs is matched to another P15 node c with |c - b| < 7 um (two predicted nodes compete for one GT cell)
  gs_unmatched_near : gs unmatched but b within 10 um of gs (localization / matching radius)
  wrong_cell: gs matched to c far (>= 7 um) from b  (a real identity error: the track jumped to another cell)
  gs_far    : gs unmatched and b >= 10 um away
  in_only   : only the target side is evaluable (b matched to GT with predecessor, a not its GT parent)"""
import sys; sys.path.insert(0, '/workspace/cl/nm')
import gr_common as C
import glob, json
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
import numpy as np
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def job(a):
    s, f = a
    import numcodecs.blosc; numcodecs.blosc.use_threads = False
    import evalx
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    full = C.load_full(FULL[s] + '/' + name + '.geff')
    U = C.build_union(nodes, edges, full)
    L = C.gt_label(name, nodes, edges, U); C.label_edges(U, L)
    gt, _ = evalx.load_gt(name); K = evalx.K
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * C.S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gsucc = {}; gpar = {}
    for (x, y) in L['GE']: gsucc.setdefault(x, []).append(y); gpar[y] = x
    g = L['g']; inp = U['inp']
    g2p = {int(g[i]): i for i in range(len(g)) if g[i] >= 0 and inp[i]}
    rpos = np.maximum(0, np.round(U['xyz'])) * C.S
    c = Counter()
    for (a, b), r in U['cand'].items():
        if not r['p15'] or not r['ev'] or r['pos']: continue
        gaa = int(g[a])
        if gaa >= 0 and gsucc.get(gaa):
            gs = gsucc[gaa][0]
            if len(gsucc[gaa]) > 1:
                # GT division: take the closer daughter
                gs = min(gsucc[gaa], key=lambda q: np.linalg.norm(gpos[q] - rpos[b]))
            cc = g2p.get(gs)
            if cc is not None:
                d = np.linalg.norm(rpos[cc] - rpos[b]); c['flip' if d < 7 else 'wrong_cell'] += 1
            else:
                d = np.linalg.norm(gpos[gs] - rpos[b]); c['gs_unmatched_near' if d < 10 else 'gs_far'] += 1
        else:
            c['in_only'] += 1
    return s, name, dict(c)


if __name__ == '__main__':
    jobs = [(s, f) for s in FULL for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    with Pool(24, maxtasksperchild=4) as p: R = p.map(job, jobs, chunksize=1)
    tot = {'44b6': Counter(), '6bba': Counter(), 'clean40': Counter()}
    for s, n, c in R:
        tot[n[:4]].update(c)
        if s in ('hold36', 'prev4'): tot['clean40'].update(c)
    for k, v in tot.items(): print(k, sum(v.values()), dict(v))
