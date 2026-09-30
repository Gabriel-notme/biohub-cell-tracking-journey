"""p19_motion: physically implausible motion in the final graph (GT-free, deletion/correction only; no edge is added).
mode='spike_del'  : zig-zag spike: node n with 1 parent p (p not dividing) and 1 child c, |x_p - x_c| < pc_max (3 um: the cell barely
                    moved over two frames) while n lies > dev_min (5 um) from the midpoint of p and c -> n is a detection of another
                    object; delete n (track gets a one-frame gap; gap edges are never scored).
mode='spike_move' : same candidates, but move n to the p/c midpoint (pure position correction, edges unchanged).
mode='out_cut'    : outlier link a->b inside a linear tracklet (>= min_len nodes): d > max(d_min, k*median step of the tracklet) and the
                    next step reverses direction (cos < 0): cut a->b.
"""
from collections import defaultdict
import numpy as np

S = np.array([1.625, .40625, .40625])


def _graph(edges):
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    return out, par


def _pos(nodes):
    return {n: np.array([float(v['z']), float(v['y']), float(v['x'])]) * S for n, v in nodes.items()}


def spikes(nodes, edges, dev_min=5.0, pc_max=3.0):
    out, par = _graph(edges)
    P = _pos(nodes)
    res = []
    for n in nodes:
        if n not in par or len(out.get(n, [])) != 1: continue
        p = par[n]; c = out[n][0]
        if len(out[p]) != 1: continue
        mid = (P[p] + P[c]) / 2
        if np.linalg.norm(P[p] - P[c]) < pc_max and np.linalg.norm(P[n] - mid) > dev_min: res.append((p, n, c))
    return res


def tracklets(nodes, out, par):
    """maximal 1-1 chains: list of node lists."""
    heads = [n for n in nodes if not (n in par and len(out[par[n]]) == 1)]
    segs = []
    for h in heads:
        ch = [h]
        while len(out.get(ch[-1], [])) == 1: ch.append(out[ch[-1]][0])
        segs.append(ch)
    return segs


def outliers(nodes, edges, d_min=6.0, k=3.0, min_len=4):
    out, par = _graph(edges)
    P = _pos(nodes)
    res = []
    for ch in tracklets(nodes, out, par):
        if len(ch) < min_len: continue
        st = [float(np.linalg.norm(P[ch[i + 1]] - P[ch[i]])) for i in range(len(ch) - 1)]
        med = float(np.median(st))
        for i in range(len(ch) - 2):  # edge ch[i]->ch[i+1] with a following step ch[i+1]->ch[i+2]
            if st[i] <= max(d_min, k * med): continue
            v1 = P[ch[i + 1]] - P[ch[i]]; v2 = P[ch[i + 2]] - P[ch[i + 1]]
            if np.dot(v1, v2) < 0: res.append((ch[i], ch[i + 1]))
    return res


def apply(nodes, edges, mode='spike_del', dev_min=5.0, pc_max=3.0, d_min=6.0, k=3.0, min_len=4, **kw):
    st = {}
    if mode in ('spike_del', 'spike_move'):
        sp = spikes(nodes, edges, dev_min, pc_max); st['mo_cand'] = len(sp)
        if not sp: return nodes, edges, st
        if mode == 'spike_del':
            gone = {n for _, n, _ in sp}
            return ({i: v for i, v in nodes.items() if i not in gone},
                    [e for e in edges if int(e['source_id']) not in gone and int(e['target_id']) not in gone], st)
        nn = dict(nodes)
        for p, n, c in sp:
            v = dict(nodes[n])
            for ax in 'zyx': v[ax] = (float(nodes[p][ax]) + float(nodes[c][ax])) / 2
            nn[n] = v
        return nn, edges, st
    if mode == 'out_cut':
        oc = set(outliers(nodes, edges, d_min, k, min_len)); st['mo_cand'] = len(oc)
        if not oc: return nodes, edges, st
        return nodes, [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in oc], st
    raise ValueError(mode)
