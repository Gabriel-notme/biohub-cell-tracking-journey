"""Consistency fix on P15: nodes added by tbext carry raw pre-ILP detection coordinates, while every node B5 kept went through
B5's linefit smoothing (weight 0.8, +-2 frames along fork-free chains). Smooth the tbext-added nodes the same way on the final
structure: new = (1 - w*0.8) * raw + w*0.8 * linefit(raw-coordinate neighbourhood); w = 1 (they never got smoothed; a-priori),
w = 0.5 as sensitivity. Neighbour inputs: raw pre-ILP coordinates for id-aligned detections, current coordinates otherwise.
Coordinates only. B5's 2 um collision rule is re-applied. GT-free."""
import sys, builtins
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/p56stage')
WANTS_META = True
S = np.array([1.625, .40625, .40625]); W = 0.8


def apply(nodes, edges, w=1.0, minsep=2.0, name=None, set=None, fullgeff=None, zarr=None):
    from scipy.spatial import cKDTree
    from edge_link import load_full
    import p15_post
    fids, fT, fV, fE, fprob = load_full(fullgeff)
    raw = {int(i): (int(t), np.asarray(v, float)) for i, t, v in zip(fids.tolist(), fT.tolist(), fV)}
    tb = builtins.set(k for k, v in nodes.items() if v.get('tb_ext'))
    if not tb: return nodes, edges, {'tbs_moved': 0}
    t_cur = {k: int(v['t']) for k, v in nodes.items()}
    pc, sc = p15_post._struct(edges, t_cur)
    orig = {}
    for k, v in nodes.items():
        f = raw.get(k); cp = np.array([v[c] for c in 'zyx'], float)
        orig[k] = f[1] if (f is not None and f[0] == int(v['t']) and np.linalg.norm((f[1] - cp) * S) <= 3.0) else cp
    have = builtins.set(orig)
    new = {k: dict(v) for k, v in nodes.items()}; changed = builtins.set()
    for k in tb:
        h = p15_post._hood(k, pc, sc, have)
        if len(h) < 3: continue
        fit = (1 - W) * orig[k] + W * p15_post._fit(h, orig)
        q = (1 - w) * np.array([nodes[k][c] for c in 'zyx'], float) + w * fit
        if np.any(q < 0) or np.any(np.rint(q) >= np.array([64, 256, 256])): continue
        new[k].update(dict(zip('zyx', map(float, q)))); changed.add(k)
    frames = defaultdict(list)
    for k, v in nodes.items(): frames[int(v['t'])].append(k)
    rej = 0
    for ns in frames.values():
        if len(ns) < 2 or not any(n in changed for n in ns): continue
        old = np.array([[nodes[n][c] for c in 'zyx'] for n in ns]) * S
        for _ in range(3):
            pp = np.array([[new[n][c] for c in 'zyx'] for n in ns]) * S
            bad = builtins.set()
            for i, j in cKDTree(pp).query_pairs(minsep):
                if np.linalg.norm(pp[i] - pp[j]) + 1e-5 < min(minsep, np.linalg.norm(old[i] - old[j])):
                    bad.update(n for n in (ns[i], ns[j]) if n in changed)
            if not bad: break
            for n in bad: new[n] = dict(nodes[n]); changed.discard(n)
            rej += len(bad)
    return new, edges, {'tbs_moved': len(changed), 'tbs_rej': rej}
