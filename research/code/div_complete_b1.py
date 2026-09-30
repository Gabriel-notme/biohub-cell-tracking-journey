import sys, os, json
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
ART = os.environ.get('BIOHUB_ART')
if ART and ART not in sys.path: sys.path.insert(0, ART)

def structure(nodes, edges):
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    return out, prev

def candidates(nodes, edges, max_pb=13.0, max_ab=20.0, min_branch=2):
    from cell_event import SCALE, chain
    out, prev = structure(nodes, edges)
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    rows = []
    for t in sorted(frames):
        nxt = frames.get(t + 1, [])
        if not nxt: continue
        tree = cKDTree(np.array([pos[n] for n in nxt]))
        for p in frames[t]:
            ch = out.get(p, [])
            if len(ch) != 1: continue
            a = ch[0]
            if len(chain(a, out, pos)) < min_branch: continue
            for j in tree.query_ball_point(pos[p], max_pb):
                b = nxt[j]
                if b == a: continue
                if np.linalg.norm(pos[a] - pos[b]) > max_ab: continue
                if len(chain(b, out, pos)) < min_branch: continue
                q = prev.get(b)
                if q is None: typ = 'start'
                elif len(out.get(q, [])) == 1 and q != p: typ = 'stolen'
                else: continue
                rows.append((p, a, b, q, typ))
    return rows, out, prev, pos

def score_candidates(er, name, nodes, edges, with_edges=False):
    from cell_event import chain, fork_geometry, edge_geometry
    rows, out, prev, pos = candidates(nodes, edges)
    if not rows: return []
    need = {x for r in rows for x in r[:3]}
    if with_edges: need |= {r[3] for r in rows if r[3] is not None}
    sub = {n: nodes[n] for n in need}
    ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
    triples = [(p, a, b) for p, a, b, q, typ in rows]
    cc = {}
    def ch(n, d, key):
        k = (n, key)
        if k not in cc: cc[k] = chain(n, d, pos)
        return cc[k]
    fg = [fork_geometry(ch(p, prev, 0), ch(a, out, 1), ch(b, out, 1)) for p, a, b in triples]
    fp = er.score('fork', triples, fg, emb, lookup)
    if with_edges:
        qb = [(q, b) for p, a, b, q, typ in rows if q is not None]
        eg = [edge_geometry(chain(q, prev, pos), chain(b, out, pos)) for q, b in qb]
        ep = er.score('edge', qb, eg, emb, lookup) if qb else []
        pb = [(p, b) for p, a, b, q, typ in rows]
        pg = [edge_geometry(chain(p, prev, pos), chain(b, out, pos)) for p, b in pb]
        pp = er.score('edge', pb, pg, emb, lookup)
    else:
        ep = [None] * len(rows); pp = [float('nan')] * len(rows)
    res = []; k = 0
    for i, (p, a, b, q, typ) in enumerate(rows):
        e_old = None
        if q is not None:
            e_old = None if ep[k] is None else float(ep[k]); k += 1
        res.append({'p': p, 'a': a, 'b': b, 'q': q, 'typ': typ, 'fork': float(fp[i]), 'e_old': e_old, 'e_pb': float(pp[i]),
                    'd_pb': float(np.linalg.norm(pos[b] - pos[p])), 'd_ab': float(np.linalg.norm(pos[a] - pos[b]))})
    return res

def apply(nodes, edges, cands, th_start=0.9, th_stolen=0.9, old_max=0.5, allow_stolen=True, allow_start=True):
    out, prev = structure(nodes, edges)
    cs = sorted(cands, key=lambda c: -c['fork'])
    used_p = set(); used_b = set(); rm = set(); add = []
    for c in cs:
        p, a, b, q = c['p'], c['a'], c['b'], c['q']
        if p in used_p or b in used_b or a in used_b: continue
        if c['typ'] == 'start':
            if not allow_start or c['fork'] < th_start: continue
        else:
            if not allow_stolen or c['fork'] < th_stolen: continue
            if old_max < 1.0 and (c['e_old'] is None or c['e_old'] > old_max): continue
            if q in used_p: continue
            rm.add((q, b)); used_p.add(q)
        used_p.add(p); used_b.add(b); used_b.add(a); add.append((p, b))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm] + [{'source_id': p, 'target_id': b, 'div_complete': 1} for p, b in add]
    return ne, {'div_added': len(add), 'div_stolen': len(rm)}

def score_candidates2(er, name, nodes, edges, max_pb=13.0, max_ab=20.0, min_branch=2):
    global candidates
    _orig = candidates
    def _c(nodes, edges, **kw): return _orig(nodes, edges, max_pb=max_pb, max_ab=max_ab, min_branch=min_branch)
    candidates = _c
    try: return score_candidates(er, name, nodes, edges)
    finally: candidates = _orig
