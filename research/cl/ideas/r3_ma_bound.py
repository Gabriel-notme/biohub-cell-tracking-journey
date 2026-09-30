"""READ-ONLY diagnostic (critic A, metric/data angle): where do P14's remaining edge errors sit with respect to data boundaries?
Per movie, official matching; every GT edge -> TP / FN class, every evaluable pred edge -> TP / FP, binned by
source frame t, GT z-slice, xy distance to the border; GT nodes -> matched / nearest-pred distance; GT divisions by t_d.
Output: /workspace/cl/ideas/r3_ma_bound_rows.json (one dict per movie)."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def job(f):
    import evalx
    from tracking_cellmot.metrics import evaluate
    from tracking_cellmot.division_metrics import score_divisions
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem
    st = [s for s in SETS if '_%s/' % s in f][0]
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    er = evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x], float) for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gtt = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs()
    gedges = [(int(s), int(d)) for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list())]
    gsucc = defaultdict(list); gpar = defaultdict(list)
    for s, d in gedges: gsucc[s].append(d); gpar[d].append(s)
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    ppos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.stack([ppos[n] for n in ns]) * S), ns) for t, ns in byt.items()}
    T = max(gtt.values()) if gtt else 99
    tmax_pred = max(int(v['t']) for v in nodes.values())

    def xyb(p):  # xy distance to border in um (image 256x256)
        return float(min(p[1], p[2], 255 - p[1], 255 - p[2]) * S[1])
    out = dict(movie=name, set=st, emb=name[:4], T=tmax_pred, gT=T, edge_tp=er.edge_tp, edge_fp=er.edge_fp, edge_fn=er.edge_fn,
               n_total=n_total, npred=len(nodes), ngt=len(gpos))
    # GT edges
    ge = []
    for s, d in gedges:
        t = gtt[s]
        ps, pd_ = g2p.get(s), g2p.get(d)
        if ps is not None and pd_ is not None and pd_ in succ.get(ps, []): c = 'tp'
        elif ps is None and pd_ is None: c = 'both_unm'
        elif ps is None: c = 'src_unm'
        elif pd_ is None: c = 'dst_unm'
        else: c = 'struct'
        ge.append((t, int(round(gpos[s][0])), round(xyb(gpos[s]), 1), c, round(float(np.linalg.norm((gpos[d] - gpos[s]) * S)), 2)))
    out['ge'] = ge
    # evaluable pred edges (FP)
    pe = []
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id'])
        gs, gd = p2g.get(s), p2g.get(d)
        ov = gs is not None and len(gsucc.get(gs, [])) > 0
        iv = gd is not None and len(gpar.get(gd, [])) > 0
        if not (ov or iv): continue
        tp = gs is not None and gd is not None and gd in gsucc.get(gs, [])
        if tp: continue
        t = int(nodes[s]['t'])
        cls = 'fp_s_unm' if gs is None else ('fp_d_unm' if gd is None else 'fp_both')
        pe.append((t, int(ppos[s][0]), round(xyb(ppos[s]), 1), cls, len(succ.get(s, [])), round(float(np.linalg.norm((ppos[d] - ppos[s]) * S)), 2)))
    out['pe'] = pe
    # GT nodes: matched? nearest pred distance
    gn = []
    for g, p in gpos.items():
        t = gtt[g]
        if g in g2p: gn.append((t, int(round(p[0])), round(xyb(p), 1), 1, round(float(np.linalg.norm((ppos[g2p[g]] - p) * S)), 2))); continue
        if t in trees:
            dd, ii = trees[t][0].query(p * S, k=1); gn.append((t, int(round(p[0])), round(xyb(p), 1), 0, round(float(dd), 2)))
        else: gn.append((t, int(round(p[0])), round(xyb(p), 1), 0, 99.0))
    out['gn'] = gn
    # divisions
    try:
        ds = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
        out['gdiv'] = [(gtt[g], v) for g, v in ds.scores.items()]
        inv2 = inv
        out['fpf'] = [int(nodes[inv2[int(x)]]['t']) for x in ds.fp_forks]
        out['tpf'] = [int(nodes[inv2[int(x)]]['t']) for x in ds.tp_forks]
    except Exception as ex:
        out['div_err'] = repr(ex)
    # pred forks by t (all)
    out['forks_t'] = Counter(int(nodes[n]['t']) for n, c in succ.items() if len(c) >= 2)
    out['nodes_t'] = Counter(int(v['t']) for v in nodes.values())
    return out


if __name__ == '__main__':
    files = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(int(os.environ.get('RULE_POOL', '40')), maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/r3_ma_bound_rows.json', 'w'))
    print('done', len(R))
