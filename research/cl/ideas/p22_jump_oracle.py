"""AUDIT ONLY (uses GT): p22_jump relinking with the per-transition shift taken from the GT median displacement (frames_an.json) at
transitions where |GT median| >= vmin, to bound what compensated relinking can recover at global jumps."""
import json, numpy as np
from collections import defaultdict
import p22_jump
WANTS_META = True
_G = None
def apply(nodes, edges, vmin=5.0, r=3.5, name=None, set=None, **kw):
    global _G
    if _G is None:
        _G = {x['movie']: {y['t']: y['gt_med'] for y in x['rows'] if y['gt_med'] is not None} for x in json.load(open('/workspace/nbrun/frames_an.json'))}
    gm = {t: np.array(v) for t, v in _G.get(name, {}).items() if np.linalg.norm(v) >= vmin}
    orig = p22_jump.est_shift
    def fake(PA, PB, R=25.0, minsup=20, _t=[None]):
        return None, 0
    # run the relinker transition by transition with the oracle shift
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    S = p22_jump.S
    t_of = {n: int(v['t']) for n, v in nodes.items()}
    pos = {n: np.array([max(0, int(round(float(v[c])))) for c in 'zyx']) * S for n, v in nodes.items()}
    byt = defaultdict(list)
    for n in nodes: byt[t_of[n]].append(n)
    E = {(int(e['source_id']), int(e['target_id'])): e for e in edges}
    ch = defaultdict(list); par = {}
    for (a, b) in E: ch[a].append(b); par[b] = a
    st = {'jump_transitions': 0, 'relinked': 0}
    for t, v in gm.items():
        if t not in byt or t + 1 not in byt: continue
        st['jump_transitions'] += 1
        A, B = byt[t], byt[t + 1]
        D = cdist(np.array([pos[n] for n in A]) + v, np.array([pos[n] for n in B])); C = np.where(D <= r, D, 1e6)
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
            E[(a, b)] = {'source_id': a, 'target_id': b}; ch[a] = [b]; par[b] = a; st['relinked'] += 1
    return nodes, list(E.values()), st
