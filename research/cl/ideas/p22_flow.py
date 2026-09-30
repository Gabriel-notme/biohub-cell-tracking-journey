"""P22 candidate: local-flow compensated relinking at large-motion transitions. For a transition t->t+1 whose median displacement over
the existing links is >= vmin um (global jumps / fast tissue flow), every node a at t gets an expected displacement v_a = component-wise
median of the existing link displacements whose source lies within rloc um of a (>= kmin links, else the transition median). Nodes are
paired by Hungarian on |p_a + v_a - p_b| <= r; for each pair (a, b) not already linked, if a has at most one child and b's parent is not
a fork parent, a's single child link and b's parent link are replaced by a->b. Forks are never created or broken. GT-free."""
import numpy as np
from collections import defaultdict
WANTS_META = True
S = np.array([1.625, 0.40625, 0.40625])


def fix(nodes, edges, vmin=5.0, rloc=20.0, r=3.0, kmin=5):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    t_of = {n: int(v['t']) for n, v in nodes.items()}
    pos = {n: np.array([max(0, int(round(float(v[c])))) for c in 'zyx']) * S for n, v in nodes.items()}
    byt = defaultdict(list)
    for n in nodes: byt[t_of[n]].append(n)
    E = {(int(e['source_id']), int(e['target_id'])): e for e in edges}
    ch = defaultdict(list); par = {}
    for (a, b) in E: ch[a].append(b); par[b] = a
    links = defaultdict(list)
    for (a, b) in E: links[t_of[a]].append((a, b))
    st = {'flow_transitions': 0, 'relinked': 0}
    for t in sorted(links):
        L = links[t]
        if t + 1 not in byt or len(L) < kmin: continue
        src = np.array([pos[a] for a, b in L]); dv = np.array([pos[b] - pos[a] for a, b in L])
        gmed = np.median(dv, 0)
        if np.linalg.norm(gmed) < vmin: continue
        st['flow_transitions'] += 1
        tree = cKDTree(src)
        A, B = byt[t], byt[t + 1]
        PA = np.array([pos[n] for n in A]); PB = np.array([pos[n] for n in B])
        V = np.empty_like(PA)
        for i, nb in enumerate(tree.query_ball_point(PA, rloc)):
            V[i] = np.median(dv[nb], 0) if len(nb) >= kmin else gmed
        D = cdist(PA + V, PB); C = np.where(D <= r, D, 1e6)
        for i, j in zip(*linear_sum_assignment(C)):
            if D[i, j] > r: continue
            a, b = A[i], B[j]
            if (a, b) in E or len(ch[a]) > 1: continue
            p = par.get(b)
            if p is not None and len(ch[p]) > 1: continue
            if ch[a]:
                c = ch[a][0]; E.pop((a, c)); ch[a] = []; par.pop(c, None)
            if p is not None:
                E.pop((p, b)); ch[p] = [x for x in ch[p] if x != b]; par.pop(b, None)
            E[(a, b)] = {'source_id': a, 'target_id': b, 'flow_relink': 1}; ch[a] = [b]; par[b] = a; st['relinked'] += 1
    return nodes, list(E.values()), st


def apply(nodes, edges, vmin=5.0, rloc=20.0, r=3.0, kmin=5, name=None, set=None, **kw):
    return fix(nodes, edges, vmin=vmin, rloc=rloc, r=r, kmin=kmin)
