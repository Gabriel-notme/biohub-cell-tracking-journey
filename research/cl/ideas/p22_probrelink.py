"""P22 candidate: at large-motion transitions, trust the learned candidate-edge probabilities (appearance-based edge model in the
pre-ILP candidate graph) instead of B5's velocity-based motion relinking. Trigger: global shift estimate (4x-downsampled 3D phase
correlation of frames t, t+1; precomputed in frames_an.json for local evaluation) with norm >= vmin um. Among nodes present in the graph
at t and t+1, candidate edges with prob >= pmin are paired by Hungarian maximising prob; for each pair (a, b) not already linked, if a
has at most one child and b's parent is not a fork parent, a's child link and b's parent link are replaced by a->b. GT-free."""
import json, numpy as np
from collections import defaultdict
WANTS_META = True
_SH = None


def shifts(name):
    global _SH
    if _SH is None:
        _SH = {r['movie']: {x['t']: x['shift'] for x in r['rows'] if not x['ident']} for r in json.load(open('/workspace/nbrun/frames_an.json'))}
    return _SH.get(name, {})


def fix(nodes, edges, fullgeff, sh, vmin=5.0, pmin=0.5, vmax=30.0):
    import sys
    sys.path.insert(0, '/workspace/p19ds')
    from edge_link import load_full
    from scipy.optimize import linear_sum_assignment
    fids, fT, fV, fE, fprob = load_full(fullgeff)
    t_of = {n: int(v['t']) for n, v in nodes.items()}
    trig = {t for t, v in sh.items() if vmin <= np.linalg.norm(v) <= vmax}
    cand = defaultdict(list)
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        a, b = int(a), int(b)
        if p >= pmin and a in nodes and b in nodes and t_of[a] in trig and t_of[b] == t_of[a] + 1: cand[t_of[a]].append((a, b, float(p)))
    E = {(int(e['source_id']), int(e['target_id'])): e for e in edges}
    ch = defaultdict(list); par = {}
    for (a, b) in E: ch[a].append(b); par[b] = a
    st = {'prob_transitions': 0, 'relinked': 0}
    for t, L in cand.items():
        st['prob_transitions'] += 1
        As = sorted({a for a, b, p in L}); Bs = sorted({b for a, b, p in L}); ia = {a: i for i, a in enumerate(As)}; ib = {b: j for j, b in enumerate(Bs)}
        C = np.full((len(As), len(Bs)), 1e6)
        for a, b, p in L: C[ia[a], ib[b]] = min(C[ia[a], ib[b]], -p)
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] > 0: continue
            a, b = As[i], Bs[j]
            if (a, b) in E or len(ch[a]) > 1: continue
            p = par.get(b)
            if p is not None and len(ch[p]) > 1: continue
            if ch[a]:
                c = ch[a][0]; E.pop((a, c)); ch[a] = []; par.pop(c, None)
            if p is not None:
                E.pop((p, b)); ch[p] = [x for x in ch[p] if x != b]; par.pop(b, None)
            E[(a, b)] = {'source_id': a, 'target_id': b, 'prob_relink': 1}; ch[a] = [b]; par[b] = a; st['relinked'] += 1
    return nodes, list(E.values()), st


def apply(nodes, edges, vmin=5.0, pmin=0.5, vmax=30.0, name=None, set=None, fullgeff=None, **kw):
    return fix(nodes, edges, fullgeff, shifts(name), vmin=vmin, pmin=pmin, vmax=vmax)
