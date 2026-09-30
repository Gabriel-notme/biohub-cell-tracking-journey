"""Crossing repair: two same-frame nodes closer than r um cannot be two distinct nuclei (annotated GT same-frame NN p1 = 7.7 um,
<0.01% below 4 um). If both are through nodes (one parent, one child, no fork involvement) and one of them deviates from its own
track's parent/child midpoint by > dmin um (and by > 2x the other's deviation), move that node to its midpoint.
mode 'move' (move deviant node to midpoint) | 'drop' (delete the deviant node; its track gets a 1-frame gap -> 2 edges lost)."""
import numpy as np
from collections import defaultdict
S = np.array([1.625, 0.40625, 0.40625])

def apply(nodes, edges, r=3.5, dmin=2.0, mode='move', ratio=2.0):
    from scipy.spatial import cKDTree
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    fpos = {n: np.array([v[k] for k in 'zyx']) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    def through(n):
        return n in par and len(out[n]) == 1 and len(out[par[n]]) == 1 and len(out[out[n][0]]) <= 1
    def dev(n):
        return float(np.linalg.norm(pos[n] - (pos[par[n]] + pos[out[n][0]]) / 2))
    moves = {}
    for t, ns in byt.items():
        if len(ns) < 2: continue
        P = np.stack([pos[n] for n in ns]); tr = cKDTree(P)
        for i, j in tr.query_pairs(r):
            a, b = ns[i], ns[j]
            if not (through(a) and through(b)): continue
            da, db = dev(a), dev(b)
            if da < db: a, b, da, db = b, a, db, da
            if da > dmin and da > ratio * db: moves[a] = True
    nodes = dict(nodes)
    if mode == 'move':
        for n in moves:
            m = (fpos[par[n]] + fpos[out[n][0]]) / 2
            nodes[n] = dict(nodes[n], z=float(m[0]), y=float(m[1]), x=float(m[2]))
    else:
        for n in moves: nodes.pop(n)
        edges = [e for e in edges if int(e['source_id']) not in moves and int(e['target_id']) not in moves]
    return nodes, edges, {'xrep': len(moves)}
