"""Shadow-segment removal (no learned parameters).
Two predicted tracks that run side by side (same-frame mutual nearest neighbours closer than D um, linked frame to frame on both sides)
for >= L frames make the official 7 um matching flip between them, costing 1 FN + up to 2 FP per flip. Within such a segment,
delete the nodes of the side whose whole fork-free track is shorter (tie: lower mean edge_prob). Forks and fork daughters are never touched."""
from collections import defaultdict
import numpy as np

S = np.array([1.625, .40625, .40625])


def segments(nodes, edges, D=5.0):
    from scipy.spatial import cKDTree
    out, par = defaultdict(list), {}
    ep = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
        try: ep[b] = float(e.get('edge_prob')) if e.get('edge_prob') is not None else 0.5
        except (TypeError, ValueError): ep[b] = 0.5
    fork = set()
    for a, ch in out.items():
        if len(ch) >= 2: fork.add(a); fork.update(ch)
    # fork-free track ids
    tid = {}; tlen = defaultdict(int); tep = defaultdict(list)
    for n in sorted(nodes, key=lambda k: int(nodes[k]['t'])):
        p = par.get(n)
        tid[n] = tid[p] if (p is not None and len(out[p]) == 1 and p in tid) else n
        tlen[tid[n]] += 1; tep[tid[n]].append(ep.get(n, 0.5))
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    mate = {}
    for t, ns in frames.items():
        if len(ns) < 2: continue
        P = np.array([pos[n] for n in ns]); dd, ii = cKDTree(P).query(P, k=2)
        for i in range(len(ns)):
            j = ii[i, 1]
            if ii[j, 1] == i and dd[i, 1] < D and ns[i] not in fork and ns[j] not in fork: mate[ns[i]] = ns[j]
    one = lambda n: out[n][0] if len(out.get(n, [])) == 1 else None
    segs, seen = [], set()
    for a in sorted(mate, key=lambda k: int(nodes[k]['t'])):
        if a in seen: continue
        b = mate[a]
        # start of a run: previous pair not a linked mate pair
        pa, pb = par.get(a), par.get(b)
        if pa is not None and pb is not None and mate.get(pa) == pb and len(out[pa]) == 1 and len(out[pb]) == 1: continue
        A, B = [a], [b]; seen.update((a, b))
        while True:
            na, nb = one(A[-1]), one(B[-1])
            if na is None or nb is None or mate.get(na) != nb: break
            A.append(na); B.append(nb); seen.update((na, nb))
        segs.append((A, B))
    return segs, tid, tlen, tep


def apply(nodes, edges, D=5.0, L=3, mode='short', matched=None):
    segs, tid, tlen, tep = segments(nodes, edges, D)
    rm = set(); nseg = 0
    for A, B in segs:
        if len(A) < L: continue
        ta, tb = tid[A[0]], tid[B[0]]
        la, lb = tlen[ta], tlen[tb]
        if mode == 'short':
            kill = B if (lb < la or (lb == la and np.mean(tep[tb]) < np.mean(tep[ta]))) else A
        elif mode == 'oracle':
            ka = sum(1 for n in A if n in matched); kb = sum(1 for n in B if n in matched)
            if ka == kb: continue
            kill = B if kb < ka else A
        elif mode == 'lowprob':
            kill = B if np.mean(tep[tb]) < np.mean(tep[ta]) else A
        rm.update(kill); nseg += 1
    nn = {k: v for k, v in nodes.items() if k not in rm}
    ne = [e for e in edges if int(e['source_id']) not in rm and int(e['target_id']) not in rm]
    return nn, ne, dict(shadow_segs=nseg, shadow_nodes=len(rm))
