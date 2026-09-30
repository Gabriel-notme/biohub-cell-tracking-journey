"""Global re-solve (gr) common code: union candidate graph (P15 nodes + dropped pre-ILP detections; P15 edges + pre-ILP edges
+ optional radius candidates), GT labelling (oracle), and a per-frame-pair assignment solver / SCIP ILP.
Only t -> t+1 edges are candidates (the metric drops all other edges)."""
import os
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
import sys
for p in ['/workspace/p56stage', '/workspace/official/src', '/workspace/code', '/workspace/cl', '/workspace/cl/nm']:
    if p not in sys.path: sys.path.insert(0, p)
from collections import defaultdict
import numpy as np

S = np.array([1.625, .40625, .40625])


def load_full(fullgeff):
    import edge_link
    return edge_link.load_full(fullgeff)


def build_union(nodes, edges, full, radius=None, knn_all=False):
    """Return dict with arrays for the union node set and the candidate edge list.
    U nodes: all P15 nodes (P15 coordinates) + pre-ILP detections whose id is not a P15 node (dropped).
    Candidate edges (dt == 1): P15 edges, pre-ILP edges, and (radius) all pairs within `radius` um
    (only between P15 nodes unless knn_all)."""
    from scipy.spatial import cKDTree
    fids, fT, fV, fE, fprob = full
    ids = list(nodes)
    t = [int(nodes[n]['t']) for n in ids]
    xyz = [[float(nodes[n][k]) for k in 'zyx'] for n in ids]
    inp = [1] * len(ids)
    have = set(ids)
    fidx = {}
    for i, fi in enumerate(fids.tolist()):
        fidx[fi] = i
        if fi in have: continue
        ids.append(fi); t.append(int(fT[i])); xyz.append([float(v) for v in fV[i]]); inp.append(0)
    ids = np.array(ids, np.int64); t = np.array(t); xyz = np.array(xyz); inp = np.array(inp, bool)
    pos = xyz * S
    ix = {int(n): i for i, n in enumerate(ids.tolist())}
    cand = {}
    for e in edges:
        a, b = ix[int(e['source_id'])], ix[int(e['target_id'])]
        if t[b] != t[a] + 1: continue
        cand[(a, b)] = {'p15': 1, 'fe': -1.0, 'src': 'p15'}
    for (a0, b0), p in zip(fE.tolist(), fprob.tolist()):
        a, b = ix.get(int(a0)), ix.get(int(b0))
        if a is None or b is None or t[b] != t[a] + 1: continue
        r = cand.setdefault((a, b), {'p15': 0, 'fe': -1.0, 'src': 'full'})
        r['fe'] = float(p)
    if radius:
        frames = defaultdict(list)
        for i in range(len(ids)):
            if knn_all or inp[i]: frames[int(t[i])].append(i)
        for tt, L in frames.items():
            if tt + 1 not in frames: continue
            R = frames[tt + 1]
            tr = cKDTree(pos[R])
            for i, js in zip(L, tr.query_ball_point(pos[L], radius)):
                for j in js:
                    cand.setdefault((i, R[j]), {'p15': 0, 'fe': -1.0, 'src': 'rad'})
    return dict(ids=ids, t=t, xyz=xyz, pos=pos, inp=inp, ix=ix, cand=cand)


def gt_label(name, nodes, edges, U):
    """Official matching of P15 nodes to GT; dropped detections are matched (max-weight 1/(1+d), 7 um, rounded px)
    only to GT nodes left unmatched by P15. Returns g (node index -> GT id or -1), GT edge set, GT out/in validity."""
    import evalx
    from scipy.optimize import linear_sum_assignment
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gid = np.array(ga[K.NODE_ID].to_list()); gT = np.array(ga['t'].to_list()); gP = np.stack([np.array(ga[k].to_list(), float) for k in 'zyx'], 1) * S
    ea = gt.edge_attrs(); GE = set(); gout = defaultdict(int); gin = defaultdict(int)
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        GE.add((int(x), int(y))); gout[int(x)] += 1; gin[int(y)] += 1
    g = -np.ones(len(U['ids']), np.int64)
    for n, gg in p2g.items(): g[U['ix'][n]] = gg
    used = set(p2g.values())
    # dropped detections -> unmatched GT nodes
    rpos = np.maximum(0, np.round(U['xyz'])) * S
    drop = np.where(~U['inp'])[0]
    free = [i for i in range(len(gid)) if int(gid[i]) not in used]
    if len(drop) and free:
        byt = defaultdict(list)
        for i in drop: byt[int(U['t'][i])].append(i)
        gbt = defaultdict(list)
        for i in free: gbt[int(gT[i])].append(i)
        for tt, gl in gbt.items():
            dl = byt.get(tt)
            if not dl: continue
            D = np.linalg.norm(rpos[dl][:, None, :] - gP[gl][None, :, :], axis=2)
            W = np.where(D <= 7.0, 1.0 / (1.0 + D), 0.0)
            r, c = linear_sum_assignment(-W)
            for a, b in zip(r, c):
                if W[a, b] > 0: g[dl[a]] = int(gid[gl[b]])
    return dict(g=g, GE=GE, gout=gout, gin=gin, n_total=n_total, n_gt_edges=len(GE))


def label_edges(U, L):
    g, GE = L['g'], L['GE']
    for (a, b), r in U['cand'].items():
        ga, gb = int(g[a]), int(g[b])
        r['pos'] = int(ga >= 0 and gb >= 0 and (ga, gb) in GE)
        r['ev'] = int((ga >= 0 and L['gout'].get(ga, 0) > 0) or (gb >= 0 and L['gin'].get(gb, 0) > 0))


def division_structure(nodes, edges):
    succ = defaultdict(list)
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id']))
    return {a: bs for a, bs in succ.items() if len(bs) >= 2}


def solve_assign(U, W, fixed_src, fixed_dst):
    """Per-frame-pair max-weight bipartite assignment (in-degree <= 1, out-degree <= 1) on candidate weights W[(a,b)] (only > 0
    are useful); nodes in fixed_src / fixed_dst are excluded (division edges kept separately). Exact when node activation is free,
    because with a fixed node set the tracking ILP separates by frame pair (appearance/disappearance costs fold into edge weights)."""
    from scipy.optimize import linear_sum_assignment
    byt = defaultdict(list)
    for (a, b), w in W.items():
        if w > 0 and a not in fixed_src and b not in fixed_dst: byt[int(U['t'][a])].append((a, b, w))
    sel = []
    for tt, L in byt.items():
        A = sorted({a for a, _, _ in L}); B = sorted({b for _, b, _ in L})
        ia = {a: i for i, a in enumerate(A)}; ib = {b: i for i, b in enumerate(B)}
        M = np.zeros((len(A), len(B)))
        for a, b, w in L: M[ia[a], ib[b]] = w
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            if M[i, j] > 0: sel.append((A[i], B[j]))
    return sel


def to_graph_dict(U, nodes, sel_edges, div_edges):
    """Build the output nodes/edges: all P15 nodes + dropped nodes used by a selected edge."""
    nn = dict(nodes)
    used = set()
    for a, b in sel_edges: used.add(a); used.add(b)
    for i in used:
        if not U['inp'][i]:
            n = int(U['ids'][i]); z, y, x = U['xyz'][i]
            nn[n] = {'node_id': n, 't': int(U['t'][i]), 'z': float(z), 'y': float(y), 'x': float(x), 'gr_added': 1}
    ne = [dict(e) for e in div_edges]
    for a, b in sel_edges:
        ne.append({'source_id': int(U['ids'][a]), 'target_id': int(U['ids'][b])})
    return nn, ne
