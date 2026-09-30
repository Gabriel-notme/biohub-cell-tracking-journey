"""Temporal-boundary re-selection (critic A probe). In the first/last K frames B5 drops ~2x more pre-ILP detections than in
the interior. For a track node n in frame t in [T0, T0+K-1] (processed inner -> outer) whose interior neighbour chain has >= 3
nodes, extrapolate the track to frame t (constant velocity from the next 3 interior nodes). If a DROPPED pre-ILP detection a in
frame t lies within r_acc um of the extrapolated point and is closer to it than n by more than `margin` um, and a has no kept node
other than n within `dup` um, move n to a's coordinates (topology, ids and node count unchanged). Mirror at the movie end.
Optionally (ext=True) also run r3_tbext edge-mode extension afterwards. GT-free."""
WANTS_META = True
import sys
from collections import defaultdict
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])


def _r(v):
    return np.array([max(0, int(round(float(x)))) for x in v], float)


def apply(nodes, edges, K=3, r_acc=3.0, margin=2.0, dup=3.5, ext=False, **meta):
    sys.path.insert(0, '/workspace/p56stage')
    from edge_link import load_full
    from scipy.spatial import cKDTree
    fids, fT, fV, fE, fprob = load_full(meta['fullgeff'])
    par = {}; succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); par[b] = a; succ[a].append(b)
    nn = {k: dict(v) for k, v in nodes.items()}
    pos = {n: _r([v['z'], v['y'], v['x']]) * S for n, v in nn.items()}
    ts = [int(v['t']) for v in nn.values()]; T0, T1 = min(ts), max(ts)
    byt = defaultdict(list)
    for n, v in nn.items(): byt[int(v['t'])].append(n)
    dix = defaultdict(list)
    for j, (i, t) in enumerate(zip(fids.tolist(), fT.tolist())):
        if int(i) not in nodes: dix[int(t)].append(j)
    moved = 0; used = set()

    def inner_chain(n, early):
        ch = []; c = n
        for _ in range(3):
            nx = succ.get(c, []) if early else ([par[c]] if c in par else [])
            if len(nx) != 1: return None
            c = nx[0]; ch.append(c)
        return ch
    frames = [(T0 + k, True) for k in reversed(range(K))] + [(T1 - k, False) for k in reversed(range(K))]
    for t, early in frames:
        if t not in dix or t not in byt: continue
        ix = dix[t]; dP = np.stack([_r(fV[j]) * S for j in ix]); dtree = cKDTree(dP)
        kP = np.stack([pos[n] for n in byt[t]]); ktree = cKDTree(kP); kn = byt[t]
        for n in byt[t]:
            ch = inner_chain(n, early)
            if ch is None: continue
            p1, p3 = pos[ch[0]], pos[ch[2]]
            v = (p1 - p3) / 2.0          # per-frame step pointing from the interior toward frame t
            pstar = p1 + v
            en = float(np.linalg.norm(pos[n] - pstar))
            if en <= margin: continue
            for d, k in zip(*dtree.query(pstar, k=min(3, len(ix)))) if len(ix) > 1 else [dtree.query(pstar, k=1)]:
                if d > r_acc or d >= en - margin: break
                j = ix[int(k)]
                if j in used: continue
                others = [kn[q] for q in ktree.query_ball_point(dP[int(k)], dup) if kn[q] != n]
                if others: continue
                used.add(j)
                nn[n]['z'], nn[n]['y'], nn[n]['x'] = float(fV[j][0]), float(fV[j][1]), float(fV[j][2])
                pos[n] = _r(fV[j]) * S; moved += 1
                break
    st = {'tb_moved': moved}
    ne = edges
    if ext:
        sys.path.insert(0, '/workspace/cl')
        from ideas import r3_tbext
        nn, ne, st2 = r3_tbext.apply(nn, ne, K=K, mode='edge', **meta); st.update(st2)
    return nn, ne, st
