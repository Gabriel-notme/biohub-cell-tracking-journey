import sys, os, json
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.spatial import cKDTree

def score_candidates(er, name, nodes, edges, max_pb=13.0, max_ab=20.0, min_branch=2):
    from cell_event import SCALE, chain, fork_geometry
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    rows = []
    for t in sorted(frames):
        starts = [n for n in frames.get(t + 1, []) if n not in prev and len(chain(n, out, pos)) >= min_branch]
        if not starts: continue
        tree = cKDTree(np.array([pos[n] for n in starts]))
        for p in frames[t]:
            if out.get(p): continue
            if p not in prev: continue
            near = sorted((starts[j] for j in tree.query_ball_point(pos[p], max_pb)), key=lambda d: float(np.linalg.norm(pos[d] - pos[p])))[:5]
            for a, b in combinations(near, 2):
                if np.linalg.norm(pos[a] - pos[b]) > max_ab: continue
                rows.append((p, a, b))
    if not rows: return []
    need = {x for r in rows for x in r}
    ids, emb = er.embeddings(name, {n: nodes[n] for n in need}); lookup = {n: i for i, n in enumerate(ids)}
    fg = [fork_geometry(chain(p, prev, pos), chain(a, out, pos), chain(b, out, pos)) for p, a, b in rows]
    fp = er.score('fork', rows, fg, emb, lookup)
    return [{'p': p, 'a': a, 'b': b, 'fork': float(f), 'da': float(np.linalg.norm(pos[a] - pos[p])), 'db': float(np.linalg.norm(pos[b] - pos[p]))} for (p, a, b), f in zip(rows, fp)]

def apply(nodes, edges, cands, th=0.9, mode='fork', prune_min_len=2):
    cs = sorted([c for c in cands if c['fork'] >= th], key=lambda c: -c['fork'])
    used = set(); add = []
    for c in cs:
        p, a, b = c['p'], c['a'], c['b']
        if used & {p, a, b}: continue
        used |= {p, a, b}
        if mode == 'fork': add += [(p, a), (p, b)]
        else: add.append((p, a if c['da'] <= c['db'] else b))
    ne = [dict(e) for e in edges] + [{'source_id': s, 'target_id': d, 'end_complete': 1} for s, d in add]
    st = {'end_added': len(add)}
    nn = nodes
    if prune_min_len:
        import prune
        nn, ne, pst = prune.prune_fragments(nodes, ne, prune_min_len); st.update(pst)
    return nn, ne, st
