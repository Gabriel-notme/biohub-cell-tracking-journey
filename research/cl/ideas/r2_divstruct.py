"""Division-structure removal rules on P14 graphs (no GT). Each rule cuts ONE daughter edge of an implausible fork
(like dfork: nodes are kept, the cut branch becomes its own track); single nodes left isolated by the cut are dropped
(what post_prune would do).
rule: 'stub'   - a daughter branch of <= k nodes that ends (no child) before the last frame: cut it (shortest one)
      'pstart' - fork parent is a track start with history <= k frames (track does not start at frame 0): cut the daughter
                 farther from the parent
      'coll'   - a daughter's first node is within r um (rounded coords) of a same-frame node outside the fork lineage: cut it
      'union'  - stub(k=2) + pstart(k=1) + coll(r=3.5)"""
import numpy as np
from collections import defaultdict
S = np.array([1.625, 0.40625, 0.40625])


def apply(nodes, edges, rule='stub', k=2, r=3.5):
    from scipy.spatial import cKDTree
    ch = defaultdict(list); par = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u
    t = {n: int(v['t']) for n, v in nodes.items()}; Tmax = max(t.values())
    pos = {n: np.array([max(0, int(round(v[c]))) for c in 'zyx']) * S for n, v in nodes.items()}

    def branch(x):
        L = [x]
        while len(ch.get(L[-1], [])) == 1: L.append(ch[L[-1]][0])
        return L

    def hist(x):
        h = 0
        while x in par and len(ch[par[x]]) == 1: x = par[x]; h += 1
        return h, (x in par), x
    byt = defaultdict(list)
    for n in nodes: byt[t[n]].append(n)
    trees = {}
    cut = set(); why = defaultdict(int)
    forks = sorted([n for n in ch if len(ch[n]) == 2], key=lambda n: (t[n], n))
    for p in forks:
        kids = ch[p]
        br = [branch(c) for c in kids]
        rules = ['stub', 'pstart', 'coll'] if rule == 'union' else [rule]
        tgt = None
        for rr in rules:
            kk = 2 if (rule == 'union' and rr == 'stub') else (1 if (rule == 'union' and rr == 'pstart') else k)
            if rr == 'stub':
                st = [i for i in (0, 1) if len(br[i]) <= kk and not ch.get(br[i][-1]) and t[br[i][-1]] < Tmax]
                if st:
                    i = min(st, key=lambda i: (len(br[i]), -np.linalg.norm(pos[kids[i]] - pos[p]))); tgt = kids[i]
            elif rr == 'pstart':
                h, fk, root = hist(p)
                if not fk and h <= kk and t[root] > 0:
                    tgt = max(kids, key=lambda c: np.linalg.norm(pos[c] - pos[p]))
            elif rr == 'coll':
                own = {p} | set(br[0]) | set(br[1])
                best = None
                for c in kids:
                    tt = t[c]
                    if tt not in trees: trees[tt] = (cKDTree(np.stack([pos[n] for n in byt[tt]])), byt[tt])
                    tr, ns = trees[tt]
                    for j in tr.query_ball_point(pos[c], r):
                        if ns[j] not in own:
                            d = float(np.linalg.norm(pos[ns[j]] - pos[c]))
                            if best is None or d < best[0]: best = (d, c)
                if best is not None: tgt = best[1]
            if tgt is not None:
                why[rr] += 1; break
        if tgt is not None: cut.add((p, tgt))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in cut]
    deg = defaultdict(int)
    for e in ne: deg[int(e['source_id'])] += 1; deg[int(e['target_id'])] += 1
    drop = {v for _, v in cut if deg[v] == 0}
    nn = {n: v for n, v in nodes.items() if n not in drop}
    st = {'cut': len(cut), 'dropped': len(drop)}
    st.update({'cut_' + a: b for a, b in why.items()})
    return nn, ne, st
