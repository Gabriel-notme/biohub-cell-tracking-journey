"""Terminal-duplicate trimming (parameter: r um, mode). A track END node a (no child) that lies within r um of a same-frame node p
which has a child (p continues) is redundant: it can supply at most one edge but can steal the official 7um match from p, which
supplies two. Delete a (and its in-edge). mode 'end' | 'start' (START node b with no parent next to a node with a parent) | 'both'.
Fork daughters are never deleted; the trimmed node's own track must have >= minlen nodes (small comps are post_prune's job)."""
import numpy as np
from collections import defaultdict
S = np.array([1.625, 0.40625, 0.40625])

def apply(nodes, edges, r=3.5, mode='end', minlen=3, iters=1, join=None):
    from scipy.spatial import cKDTree
    nodes = dict(nodes); edges = list(edges)
    total = 0
    for it in range(iters):
        out = defaultdict(list); par = {}
        for e in edges:
            a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
        pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
        byt = defaultdict(list)
        for n, v in nodes.items(): byt[int(v['t'])].append(n)
        def tlen_back(n):
            L = 1
            while n in par: n = par[n]; L += 1
            return L
        def tlen_fwd(n):
            L = 1
            while len(out[n]) == 1: n = out[n][0]; L += 1
            return L
        drop = set(); add = []
        for t, ns in byt.items():
            if len(ns) < 2: continue
            P = np.stack([pos[n] for n in ns]); tr = cKDTree(P)
            for i, n in enumerate(ns):
                is_end = (not out[n]) and n in par
                is_start = (n not in par) and len(out[n]) == 1
                if not ((is_end and mode in ('end', 'both')) or (is_start and mode in ('start', 'both'))): continue
                if is_end and (len(out.get(par[n], [])) == 2): continue  # fork daughter
                if is_end and tlen_back(n) < minlen: continue
                if is_start and tlen_fwd(n) < minlen: continue
                for j in tr.query_ball_point(pos[n], r):
                    m = ns[j]
                    if m == n or m in drop: continue
                    if is_end and out[m]:
                        if join and m not in par and len(out[m]) == 1 and it == 0:
                            if join == 'keep_end': drop.add(m); add.append((n, out[m][0]))
                            else: drop.add(n); add.append((par[n], m))
                        else: drop.add(n)
                        break
                    if is_start and m in par: drop.add(n); break
        if not drop: break
        total += len(drop)
        for n in drop: nodes.pop(n, None)
        edges = [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
        edges += [{'source_id': a, 'target_id': b, 'dup_join': 1} for a, b in add if a not in drop and b not in drop]
    return nodes, edges, {'trim': total}
