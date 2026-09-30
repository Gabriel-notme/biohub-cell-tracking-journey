"""Per-tracklet table for B5 graphs: structure/geometry features + how many GT-matched nodes / TP edges each tracklet carries.
Output /workspace/cl/trk/<set>__<movie>.npz. Used to test whether GT (annotated lineages) can be separated from the rest,
since every predicted node off the annotated lineages costs 0.1*J/N_total in the adjusted Jaccard."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
SETS = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
        'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
        't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}
OUT = Path('/workspace/cl/trk'); OUT.mkdir(exist_ok=True)


def job(args):
    s, f = args
    name = Path(f).stem
    of = OUT / ('%s__%s.npz' % (s, name))
    if of.exists(): return name
    if not Path('/workspace/data/train/%s.geff/zarr.json' % name).exists(): return None
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    try:
        gt, n_total = evalx.load_gt(name)
    except Exception:
        return None
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); gedge = set(zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()))
    ch = defaultdict(list); par = {}; ep = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a; ep[b] = float(e['edge_prob']) if e.get('edge_prob') is not None else np.nan
    # tracklets: maximal chains where each internal node has exactly one child and child has this as only parent
    heads = [n for n in nodes if (n not in par) or len(ch[par[n]]) != 1]
    rows = []
    comp = {}
    # component id via union over edges
    root_of = {}
    for n in nodes:
        r = n
        while r in par: r = par[r]
        root_of[n] = r
    comp_size = defaultdict(int)
    for n in nodes: comp_size[root_of[n]] += 1
    comp_tspan = defaultdict(lambda: [999, -1])
    for n in nodes:
        c = comp_tspan[root_of[n]]; t = nodes[n]['t']; c[0] = min(c[0], t); c[1] = max(c[1], t)
    for h in heads:
        chain = [h]
        while len(ch[chain[-1]]) == 1: chain.append(ch[chain[-1]][0])
        P = np.array([[nodes[n]['t'], nodes[n]['z'], nodes[n]['y'], nodes[n]['x']] for n in chain], float)
        nm = sum(1 for n in chain if n in p2g)
        tp = 0
        seq = ([par[h]] if h in par else []) + chain
        for a, b in zip(seq[:-1], seq[1:]):
            if a in p2g and b in p2g and (p2g[a], p2g[b]) in gedge: tp += 1
        st = 0 if h not in par else (2 if len(ch[par[h]]) >= 2 else 1)
        en = len(ch[chain[-1]])  # 0 end, 2 division
        sc = [ep[n] for n in chain if n in ep]
        v = np.linalg.norm(np.diff(P[:, 1:] * np.array([1.625, 0.40625, 0.40625]), axis=0), axis=1) if len(chain) > 1 else np.array([0.])
        c = root_of[h]
        rows.append([len(chain), P[0, 0], P[-1, 0], st, en, *P[:, 1:].mean(0), *P[:, 1:].std(0), float(np.mean(v)), float(np.max(v)),
                     comp_size[c], comp_tspan[c][0], comp_tspan[c][1], float(np.nanmean(sc)) if sc else -1, nm, tp])
    X = np.array(rows, float)
    np.savez(of, X=X, n_total=n_total, n_pred=len(nodes), n_gt=gt.num_nodes(), n_gt_edges=len(gedge))
    return name


if __name__ == '__main__':
    jobs = [(s, f) for s, d in SETS.items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 48) as p: R = p.map(job, jobs)
    print('done', sum(r is not None for r in R), 'of', len(jobs))
