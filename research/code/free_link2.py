import sys, os, json
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
ART = os.environ.get('BIOHUB_ART', '/workspace/art_b56/artifact_bundle')
if ART not in sys.path: sys.path.insert(0, ART)

def structure(nodes, edges):
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    return out, prev

def hist_len(n, prev, cap=30):
    c = 0
    while n in prev and c < cap: n = prev[n]; c += 1
    return c

def fut_len(n, out, cap=30):
    c = 0
    while len(out.get(n, [])) == 1 and c < cap: n = out[n][0]; c += 1
    return c

def score_candidates(er, name, nodes, edges, dmax=12.0):
    from cell_event import SCALE, chain, edge_geometry
    out, prev = structure(nodes, edges)
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    T = max(int(v['t']) for v in nodes.values())
    ends = defaultdict(list); starts = defaultdict(list)
    for n, v in nodes.items():
        t = int(v['t'])
        if t < T and len(out.get(n, [])) == 0: ends[t].append(n)
        if t > 0 and n not in prev: starts[t].append(n)
    pairs = []
    for t, E in ends.items():
        S = starts.get(t + 1, [])
        if not S: continue
        tree = cKDTree(np.array([pos[b] for b in S]))
        for a in E:
            for j in tree.query_ball_point(pos[a], dmax): pairs.append((a, S[j]))
    if not pairs: return []
    need = {x for ab in pairs for x in ab}
    sub = {n: nodes[n] for n in need}
    ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
    geom = [edge_geometry(chain(a, prev, pos), chain(b, out, pos)) for a, b in pairs]
    pr = er.score('edge', pairs, geom, emb, lookup)
    res = []
    for (a, b), p in zip(pairs, pr):
        res.append({'a': a, 'b': b, 't': int(nodes[a]['t']), 'p': float(p), 'd': float(np.linalg.norm(pos[b] - pos[a])),
                    'ha': hist_len(a, prev), 'fb': fut_len(b, out)})
    return res

def apply(nodes, edges, cands, th=0.95, dmax=8.0, min_ha=0, min_fb=0):
    byt = defaultdict(list)
    for c in cands:
        if c['p'] >= th and c['d'] <= dmax and c['ha'] >= min_ha and c['fb'] >= min_fb: byt[c['t']].append(c)
    add = []
    for t, cs in byt.items():
        A = sorted({c['a'] for c in cs}); B = sorted({c['b'] for c in cs})
        ia = {a: i for i, a in enumerate(A)}; ib = {b: j for j, b in enumerate(B)}
        C = np.full((len(A), len(B)), 1e6)
        for c in cs: C[ia[c['a']], ib[c['b']]] = -np.log(max(c['p'], 1e-6)) + 0.01 * c['d']
        r, k = linear_sum_assignment(C)
        for i, j in zip(r, k):
            if C[i, j] < 1e5: add.append((A[i], B[j]))
    ne = [dict(e) for e in edges] + [{'source_id': a, 'target_id': b, 'free_link': 1} for a, b in add]
    return ne, {'free_links': len(add)}
