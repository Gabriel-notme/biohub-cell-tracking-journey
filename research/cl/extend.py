"""Track-end extension: continue free track ends forward (and free track starts backward) with extrapolated positions for up to K frames,
stopping when any existing predicted node is within R um of the extrapolated position. Adding a node costs only ~1/1600 of a TP edge
(node-count multiplier), while an extension that follows a GT cell recovers TP edges.
Pure-geometry, no images. apply(nodes, edges, T, K_fwd, K_bwd, R, vel) -> nodes, edges, stats"""
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])


def apply(nodes, edges, T=None, K_fwd=3, K_bwd=0, R=7.0, vel=0.0, nvel=3, min_len=3, border_um=0.0, shape=None):
    nodes = dict(nodes); edges = list(edges)
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    ts = sorted({int(v['t']) for v in nodes.values()}); tmin, tmax = ts[0], ts[-1]
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([[nodes[n]['z'], nodes[n]['y'], nodes[n]['x']] for n in ns]) * S) for t, ns in frames.items()}
    added = defaultdict(list)  # t -> positions (um) of added nodes, to avoid two extensions colliding
    nid = max(nodes) + 1
    st = dict(ext_fwd_nodes=0, ext_bwd_nodes=0, ext_fwd_tracks=0, ext_bwd_tracks=0)

    def chain_back(n, k):
        c = [n]
        while len(c) < k and c[-1] in par: c.append(par[c[-1]])
        return c

    def chain_fwd(n, k):
        c = [n]
        while len(c) < k and len(ch.get(c[-1], [])) == 1: c.append(ch[c[-1]][0])
        return c

    def blocked(t, p):
        if t in trees and trees[t].query_ball_point(p, R): return True
        for q in added.get(t, []):
            if np.linalg.norm(q - p) < R: return True
        return False

    def inside(p):
        if shape is None or border_um <= 0: return True
        lim = np.array(shape) * S
        return bool(np.all(p >= border_um) and np.all(p <= lim - border_um))
    new_nodes = {}; new_edges = []
    ends = [n for n in list(nodes) if not ch.get(n) and int(nodes[n]['t']) < tmax]
    starts = [n for n in list(nodes) if n not in par and int(nodes[n]['t']) > tmin]
    for dirn, lst, K in [(1, ends, K_fwd), (-1, starts, K_bwd)]:
        if K <= 0: continue
        for n in lst:
            c = chain_back(n, nvel) if dirn == 1 else chain_fwd(n, nvel)
            if len(c) < min_len and len((chain_back if dirn == 1 else chain_fwd)(n, min_len)) < min_len: continue
            P = np.array([[nodes[x]['z'], nodes[x]['y'], nodes[x]['x']] for x in c]) * S
            v = (P[0] - P[-1]) / max(1, len(c) - 1) * vel if len(c) > 1 else np.zeros(3)
            prev = n; t0 = int(nodes[n]['t']); k_done = 0
            for j in range(1, K + 1):
                t = t0 + dirn * j
                if t < tmin or t > tmax: break
                p = P[0] + v * j
                if blocked(t, p) or not inside(p): break
                new_nodes[nid] = {'t': t, 'z': float(p[0] / S[0]), 'y': float(p[1] / S[1]), 'x': float(p[2] / S[2]), 'ext': 1}
                added[t].append(p)
                new_edges.append({'source_id': prev, 'target_id': nid, 'ext': 1} if dirn == 1 else {'source_id': nid, 'target_id': prev, 'ext': 1})
                prev = nid; nid += 1; k_done += 1
            if k_done:
                st['ext_fwd_tracks' if dirn == 1 else 'ext_bwd_tracks'] += 1; st['ext_fwd_nodes' if dirn == 1 else 'ext_bwd_nodes'] += k_done
    nodes.update(new_nodes)
    return nodes, edges + new_edges, st
