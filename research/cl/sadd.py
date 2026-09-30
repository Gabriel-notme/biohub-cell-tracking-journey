"""Standalone re-insertion of dropped pre-ILP detection chains (never connected to existing tracks).
Chains follow mutual-best candidate-graph edges among dropped detections that are clear of every kept node by >= clear um."""
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])


def chains(nodes, full, clear=5.0, pmin=0.5):
    fids, fT, fV, fE, fprob = full
    idx = {int(i): j for j, i in enumerate(fids.tolist())}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([[nodes[n][k] for k in 'zyx'] for n in ns]) * S) for t, ns in frames.items()}
    elig = set()
    for j, f in enumerate(fids.tolist()):
        if f in nodes: continue
        t = int(fT[j])
        if t in trees and trees[t].query(fV[j] * S)[0] < clear: continue
        elig.add(int(f))
    bout = {}; bin_ = {}
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        if a in elig and b in elig and p >= pmin:
            if p > bout.get(a, (None, -1))[1]: bout[a] = (b, p)
            if p > bin_.get(b, (None, -1))[1]: bin_[b] = (a, p)
    nxt = {a: b for a, (b, p) in bout.items() if bin_.get(b, (None,))[0] == a}
    prv = {b: a for a, b in nxt.items()}
    out = []
    for a in elig:
        if a in prv: continue
        c = [a]
        while c[-1] in nxt: c.append(nxt[c[-1]])
        out.append(c)
    return out, idx


def apply(nodes, edges, full, clear=5.0, pmin=0.5, min_len=3, max_len=1000):
    fids, fT, fV, fE, fprob = full
    ch, idx = chains(nodes, full, clear, pmin)
    nodes = dict(nodes); edges = list(edges); k = n = 0
    for c in ch:
        if len(c) < min_len or len(c) > max_len: continue
        if any(x in nodes for x in c): continue
        for x in c:
            j = idx[x]; nodes[x] = {'node_id': x, 't': int(fT[j]), 'z': float(fV[j][0]), 'y': float(fV[j][1]), 'x': float(fV[j][2]), 'sadd': 1}
        edges += [{'source_id': a, 'target_id': b, 'sadd': 1} for a, b in zip(c[:-1], c[1:])]
        k += 1; n += len(c)
    return nodes, edges, {'sadd_chains': k, 'sadd_nodes': n}
