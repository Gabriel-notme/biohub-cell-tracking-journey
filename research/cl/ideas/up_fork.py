"""UPSTREAM lens (read-only): pre-ILP candidate forks (fullgraph nodes with 2 candidate children) vs P13 and GT divisions.
For each candidate fork u->{c1,c2}: P13 status, probs, GT label (u's GT match, or nearest GT node within 7um, is a dividing GT node
+-1 frame => 'P'; matched GT node has children but no division within +-1 => 'N' (would be an evaluable FP fork); else 'U').
Also: for each GT division, is there a candidate fork near its parent (+-1 frame)?"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def job(a):
    s, f = a
    import evalx, zarr
    from tracking_cellmot.metrics import evaluate
    import tracking_cellmot.division_metrics as DM
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    g = zarr.open_group(FULL[s] + '/' + name + '.geff', mode='r')
    fid = np.asarray(g['nodes/ids'][:]).astype(np.int64)
    fP = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'tzyx'], 1)
    fE = np.asarray(g['edges/ids'][:]).astype(np.int64); fp = np.asarray(g['edges/props/edge_prob/values'][:])
    fidx = {int(i): j for j, i in enumerate(fid.tolist())}
    cch = defaultdict(list); cprob = {}; cpar = {}
    for (u, v), p in zip(fE.tolist(), fp.tolist()):
        cch[u].append(v); cprob[(u, v)] = p; cpar[v] = u
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gid = np.array(ga[K.NODE_ID].to_list()); gT = np.array(ga['t'].to_list()); gX = np.stack([np.array(ga[k].to_list()) for k in 'zyx'], 1) * S
    ea = gt.edge_attrs(); gch = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gch[int(x)].append(int(y)); gpar[int(y)] = int(x)
    gdiv = {n for n, c in gch.items() if len(c) >= 2}
    def near_div(gn):  # gn or its GT parent/child (+-1 frame) is dividing
        if gn in gdiv: return True
        if gn in gpar and gpar[gn] in gdiv: return True
        return any(c in gdiv for c in gch.get(gn, []))
    trees = {}
    for t in np.unique(gT): k = gT == t; trees[int(t)] = (cKDTree(gX[k]), gid[k])
    def gmatch(u):
        if u in nodes: return p2g.get(u)
        j = fidx[u]; t = int(fP[j, 0])
        if t not in trees: return None
        d, i = trees[t][0].query(fP[j, 1:] * S)
        return int(trees[t][1][i]) if d <= 7. else None
    ch = defaultdict(list); par = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u
    res = DM.score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tpf = {inv[x] for x in res.tp_forks}; fpf = {inv[x] for x in res.fp_forks}
    # P13 fork near u (u itself, its P13 parent, or its P13 child)
    def p13_fork_near(u):
        if u not in nodes: return None
        c = [u] + ([par[u]] if u in par else []) + list(ch.get(u, []))
        for x in c:
            if len(ch.get(x, [])) >= 2: return 'TP' if x in tpf else ('FP' if x in fpf else 'U')
        return None
    rows = []
    for u, cs in cch.items():
        if len(cs) < 2: continue
        cs = sorted(cs, key=lambda c: -cprob[(u, c)])[:2]
        st = []
        for c in cs:
            if c not in nodes: st.append('drop')
            elif u in nodes and par.get(c) == u: st.append('linked')
            elif c not in par: st.append('start')
            else: st.append('other_par')
        gn = gmatch(u)
        if gn is None: lab = 'U'
        elif near_div(gn): lab = 'P'
        elif gch.get(gn): lab = 'N'
        else: lab = 'U'
        rows.append({'u_in': int(u in nodes), 'st': st, 'p': [cprob[(u, c)] for c in cs], 'lab': lab, 'p13': p13_fork_near(u),
                     't': int(fP[fidx[u], 0]), 'sis': float(np.linalg.norm((fP[fidx[cs[0]], 1:] - fP[fidx[cs[1]], 1:]) * S))})
    # GT divisions: candidate fork near the parent?
    cf_nodes = [u for u, cs in cch.items() if len(cs) >= 2]
    cfX = np.array([fP[fidx[u]] for u in cf_nodes]) if cf_nodes else np.zeros((0, 4))
    gd = []
    for d in gdiv:
        j = int(np.where(gid == d)[0][0]); t = gT[j]
        k = np.abs(cfX[:, 0] - t) <= 1 if len(cfX) else np.zeros(0, bool)
        dm = float(np.min(np.linalg.norm(cfX[k, 1:] * S - gX[j], axis=1))) if k.sum() else 99.
        gd.append({'dmin_candfork': dm})
    return {'set': s, 'movie': name, 'rows': rows, 'gd': gd}


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(60) as p: R = p.map(job, jobs, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/up_fork_rows.json', 'w'))
    C = defaultdict(Counter)
    for r in R:
        e = r['movie'][:4]; cl = r['set'] in ('hold36', 'prev4')
        for x in r['rows']:
            k = (tuple(sorted(x['st'])), 'p13fork' if x['p13'] else 'no_p13fork', 'minp>=.8' if min(x['p']) >= .8 else 'minp<.8')
            C[k][(e, x['lab'])] += 1
            if cl: C[k][('clean', x['lab'])] += 1
    for k in sorted(C, key=lambda k: -sum(C[k].values())):
        c = C[k]
        print('%-60s' % str(k), ' | '.join('%s P %3d N %3d U %5d' % (g, c[(g, 'P')], c[(g, 'N')], c[(g, 'U')]) for g in ['44b6', '6bba', 'clean']))
    D = Counter()
    for r in R:
        for x in r['gd']: D[(r['movie'][:4], x['dmin_candfork'] <= 7)] += 1
    print('GT divisions with a candidate fork within 7um +-1 frame:', dict(D))
