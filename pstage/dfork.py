"""Double-fork resolution: a cell cannot divide twice within K frames. When a fork has another
fork in one of its branches within K frames, the earlier fork keeps only the branch that leads to
the later fork."""
from collections import defaultdict

def resolve(nodes, edges, K=2):
    succ = defaultdict(list)
    for e in edges:
        succ[int(e['source_id'])].append(int(e['target_id']))
    forks = {n for n in nodes if len(succ.get(n, [])) >= 2}
    remove = set()
    for f in sorted(forks, key=lambda n: (nodes[n]['t'], n)):
        if len(succ[f]) != 2: continue
        hit = None
        for c in succ[f]:
            fr = [(c, 1)]
            while fr and hit is None:
                x, dd = fr.pop()
                if x in forks: hit = c; break
                if dd < K:
                    for y in succ.get(x, []): fr.append((y, dd + 1))
            if hit is not None: break
        if hit is None: continue
        other = [c for c in succ[f] if c != hit][0]
        remove.add((f, other))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in remove]
    return ne, {'dfork_removed': len(remove)}
