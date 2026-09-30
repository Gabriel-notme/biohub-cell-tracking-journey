"""P22 candidate: frozen-frame consistency. When raw frames t and t+1 are bit-identical (acquisition repeated the volume; GT nodes are
identical across such pairs), the tracking across t->t+1 must be the identity. Nodes at t and t+1 are paired by Hungarian matching on
rounded-coordinate distance (<= r um); for every pair (a, b): a's other edges into t+1 and b's other incoming edge are dropped and a->b is
added (so no fork can sit on a frozen transition). Unpaired nodes are left alone. GT-free; identical frames come from the image
(precomputed in /workspace/nbrun/frames_an.json for local evaluation)."""
import json
import numpy as np
from collections import defaultdict
WANTS_META = True
S = np.array([1.625, 0.40625, 0.40625])
_ID = None


def ident_transitions(name):
    global _ID
    if _ID is None:
        _ID = {r['movie']: [x['t'] for x in r['rows'] if x['ident']] for r in json.load(open('/workspace/nbrun/frames_an.json'))}
    return _ID.get(name, [])


def fix(nodes, edges, idents, r=3.0):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    t_of = {n: int(v['t']) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n in nodes: byt[t_of[n]].append(n)
    E = {(int(e['source_id']), int(e['target_id'])): e for e in edges}
    st = {'pairs': 0, 'added': 0, 'removed': 0, 'transitions': 0}
    for t in sorted(idents):
        A, B = byt.get(t, []), byt.get(t + 1, [])
        if not A or not B: continue
        st['transitions'] += 1
        PA = np.array([[max(0, int(round(float(nodes[n][c])))) for c in 'zyx'] for n in A]) * S
        PB = np.array([[max(0, int(round(float(nodes[n][c])))) for c in 'zyx'] for n in B]) * S
        D = cdist(PA, PB); C = np.where(D <= r, D, 1e6)
        ia, ib = linear_sum_assignment(C)
        pairs = [(A[i], B[j]) for i, j in zip(ia, ib) if D[i, j] <= r]
        out_t = defaultdict(list); in_t1 = {}
        for (s_, d_) in E:
            if t_of[s_] == t and t_of[d_] == t + 1: out_t[s_].append(d_); in_t1[d_] = s_
        for a, b in pairs:
            st['pairs'] += 1
            for d_ in list(out_t.get(a, [])):
                if d_ != b and (a, d_) in E: E.pop((a, d_)); st['removed'] += 1
            s_ = in_t1.get(b)
            if s_ is not None and s_ != a and (s_, b) in E: E.pop((s_, b)); st['removed'] += 1
            if (a, b) not in E:
                E[(a, b)] = {'source_id': a, 'target_id': b, 'frozen_link': 1}; st['added'] += 1
    return nodes, list(E.values()), st


def apply(nodes, edges, r=3.0, name=None, set=None, **kw):
    return fix(nodes, edges, ident_transitions(name), r=r)
