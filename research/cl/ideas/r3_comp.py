"""Remove isolated weakly connected components of exactly `size` nodes (post_prune 2 only removes single nodes).
near: None = all such components; float = only if every node of the component has another node within `near` um in its frame."""
from collections import defaultdict
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])


def apply(nodes, edges, size=2, near=None):
    from scipy.spatial import cKDTree
    adj = defaultdict(set)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); adj[a].add(b); adj[b].add(a)
    seen = set(); comps = []
    for n in nodes:
        if n in seen: continue
        st = [n]; seen.add(n); c = []
        while st:
            x = st.pop(); c.append(x)
            for y in adj[x]:
                if y not in seen: seen.add(y); st.append(y)
        if len(c) == size: comps.append(c)
    if near is not None and comps:
        byt = defaultdict(list)
        for n, v in nodes.items(): byt[int(v['t'])].append(n)
        pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
        trees = {t: cKDTree(np.stack([pos[n] for n in ns])) for t, ns in byt.items()}
        def has_near(n):
            t = int(nodes[n]['t']); return len(trees[t].query_ball_point(pos[n], near)) >= 2
        comps = [c for c in comps if all(has_near(n) for n in c)]
    drop = {n for c in comps for n in c}
    return {k: v for k, v in nodes.items() if k not in drop}, [e for e in edges if int(e['source_id']) not in drop], {'comp_removed': len(comps)}
