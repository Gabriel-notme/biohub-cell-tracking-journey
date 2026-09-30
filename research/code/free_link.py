import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from swap_repair import build, back_vel, fwd_vel

def link_free_ends(nodes, edges, dmax=6.0, cmax=16.0, k=2, min_hist=0, min_fut=0, motion=True):
    edges = [dict(e) for e in edges]
    succ, par, pos = build(nodes, edges)
    T = max(int(v['t']) for v in nodes.values())
    ends = defaultdict(list); starts = defaultdict(list)
    for n, v in nodes.items():
        t = int(v['t'])
        if t < T and len(succ.get(n, [])) == 0: ends[t].append(n)
        if t > 0 and n not in par: starts[t].append(n)
    def hist_len(n):
        c = 0
        while n in par and c < 50: n = par[n]; c += 1
        return c
    def fut_len(n):
        c = 0
        while len(succ.get(n, [])) == 1 and c < 50: n = succ[n][0]; c += 1
        return c
    added = []
    for t in sorted(ends):
        E = [a for a in ends[t] if hist_len(a) >= min_hist]; S = [b for b in starts.get(t + 1, []) if fut_len(b) >= min_fut]
        if not E or not S: continue
        C = np.full((len(E), len(S)), 1e6)
        for i, a in enumerate(E):
            va = back_vel(a, par, pos, k) if motion else None
            for j, b in enumerate(S):
                d = pos[b] - pos[a]; raw = float(np.linalg.norm(d))
                if raw > dmax: continue
                vb = fwd_vel(b, succ, pos, k) if motion else None
                c = raw * raw
                if va is not None: c = min(c, float((d - va) @ (d - va)))
                if vb is not None: c = 0.5 * c + 0.5 * min(raw * raw, float((d - vb) @ (d - vb)))
                C[i, j] = c
        r, cidx = linear_sum_assignment(C)
        for i, j in zip(r, cidx):
            if C[i, j] <= cmax: added.append((E[i], S[j]))
    edges += [{'source_id': a, 'target_id': b, 'free_link': 1} for a, b in added]
    return edges, {'free_links': len(added)}
