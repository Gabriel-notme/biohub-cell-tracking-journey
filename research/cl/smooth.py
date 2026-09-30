"""Temporal smoothing of node coordinates along fork-free tracks (no learned parameters).
mode: axes ('zyx' | 'yx' | 'z'); w: half-window weights, e.g. (0.5, 0.25) = [.25,.5,.25]. Only interior nodes whose neighbours
on both sides exist within the same fork-free chain are moved; forks, daughters and track ends keep their coordinates."""
from collections import defaultdict


def apply(nodes, edges, axes='zyx', w=(0.5, 0.25)):
    out, par = defaultdict(list), {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    nxt = lambda n: out[n][0] if len(out.get(n, [])) == 1 else None
    prv = lambda n: par[n] if (n in par and len(out[par[n]]) == 1) else None
    k = len(w) - 1
    nn = {}
    moved = 0
    for n, v in nodes.items():
        L, R = [], []
        c = n
        for _ in range(k):
            c = prv(c) if c is not None else None
            L.append(c)
        c = n
        for _ in range(k):
            c = nxt(c) if c is not None else None
            R.append(c)
        if any(x is None for x in L + R) or len(out.get(n, [])) != 1:
            nn[n] = v; continue
        d = dict(v)
        for ax in axes:
            s = w[0] * float(v[ax]) + sum(w[i + 1] * (float(nodes[L[i]][ax]) + float(nodes[R[i]][ax])) for i in range(k))
            d[ax] = s
        nn[n] = d; moved += 1
    return nn, edges, dict(smoothed=moved)
