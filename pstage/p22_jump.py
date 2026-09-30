"""P22 candidate: global-jump compensated relinking. Some acquisitions move the whole field between two frames (GT median displacement
up to ~10 um); our linking there has 2-5x the normal FN rate and many FP links. For each transition t->t+1 the global shift v is the mode
of all node displacement vectors within 25 um (3D histogram, 1 um bins, 3x3x3 smoothing, refined by the mean of vectors within 1.5 um of
the mode; needs >= minsup supporting pairs). Where |v| >= vmin, nodes are paired by Hungarian on |p_t + v - p_t+1| <= r; for each pair
(a, b) that is not already linked: if a has at most one child and b's current parent is not a fork parent, a's single child link and b's
parent link are replaced by a->b. Forks are never created or broken. GT-free."""
import numpy as np
from collections import defaultdict
WANTS_META = True
S = np.array([1.625, 0.40625, 0.40625])


def est_shift(PA, PB, R=25.0, minsup=20):
    from scipy.spatial import cKDTree
    from scipy.ndimage import uniform_filter
    tb = cKDTree(PB); vec = []
    for i, nb in enumerate(tb.query_ball_point(PA, R)):
        if nb: vec.append(PB[nb] - PA[i])
    if not vec: return None, 0
    V = np.concatenate(vec, 0)
    H, edges = np.histogramdd(V, bins=(50, 50, 50), range=((-R, R),) * 3)
    H = uniform_filter(H, 3, mode='constant')
    i = np.unravel_index(np.argmax(H), H.shape)
    c = np.array([(edges[k][i[k]] + edges[k][i[k] + 1]) / 2 for k in range(3)])
    sel = np.linalg.norm(V - c, axis=1) <= 1.5
    if sel.sum() < minsup: return None, int(sel.sum())
    return V[sel].mean(0), int(sel.sum())


def fix(nodes, edges, vmin=6.0, r=3.0, minsup=20, placebo=0):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    t_of = {n: int(v['t']) for n, v in nodes.items()}
    pos = {n: np.array([max(0, int(round(float(v[c])))) for c in 'zyx']) * S for n, v in nodes.items()}
    byt = defaultdict(list)
    for n in nodes: byt[t_of[n]].append(n)
    E = {(int(e['source_id']), int(e['target_id'])): e for e in edges}
    ch = defaultdict(list); par = {}
    for (a, b) in E: ch[a].append(b); par[b] = a
    st = {'jump_transitions': 0, 'relinked': 0, 'pairs': 0}
    shifts = {}
    for t in sorted(byt):
        if t + 1 not in byt: continue
        A, B = byt[t], byt[t + 1]
        PA = np.array([pos[n] for n in A]); PB = np.array([pos[n] for n in B])
        v, sup = est_shift(PA, PB, minsup=minsup)
        if v is None or np.linalg.norm(v) < vmin: continue
        st['jump_transitions'] += 1; shifts[t] = v.tolist()
        if placebo:
            rg = np.random.default_rng(placebo * 100003 + t); u = rg.normal(size=3); v = u / np.linalg.norm(u) * np.linalg.norm(v)
        D = cdist(PA + v, PB); C = np.where(D <= r, D, 1e6)
        ia, ib = linear_sum_assignment(C)
        for i, j in zip(ia, ib):
            if D[i, j] > r: continue
            a, b = A[i], B[j]; st['pairs'] += 1
            if (a, b) in E: continue
            if len(ch[a]) > 1: continue
            p = par.get(b)
            if p is not None and len(ch[p]) > 1: continue
            if ch[a]:
                c = ch[a][0]; E.pop((a, c)); ch[a] = []; par.pop(c, None)
            if p is not None:
                E.pop((p, b)); ch[p] = [x for x in ch[p] if x != b]; par.pop(b, None)
            E[(a, b)] = {'source_id': a, 'target_id': b, 'jump_relink': 1}; ch[a] = [b]; par[b] = a; st['relinked'] += 1
    return nodes, list(E.values()), st


def apply(nodes, edges, vmin=6.0, r=3.0, minsup=20, placebo=0, name=None, set=None, **kw):
    return fix(nodes, edges, vmin=vmin, r=r, minsup=minsup, placebo=placebo)
