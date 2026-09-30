import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
SCALE = np.array([1.625, 0.40625, 0.40625])

def build(nodes, edges):
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * SCALE for n, v in nodes.items()}
    return succ, par, pos

def back_vel(n, par, pos, k):
    pts = [pos[n]]; c = n
    for _ in range(k):
        p = par.get(c)
        if p is None: break
        pts.append(pos[p]); c = p
    if len(pts) < 2: return None
    return (pts[0] - pts[-1]) / (len(pts) - 1)

def fwd_vel(n, succ, pos, k):
    pts = [pos[n]]; c = n
    for _ in range(k):
        ch = succ.get(c, [])
        if len(ch) != 1: break
        c = ch[0]; pts.append(pos[c])
    if len(pts) < 2: return None
    return (pts[-1] - pts[0]) / (len(pts) - 1)

def jcost(a, b, succ, par, pos, k, w_disp):
    d = pos[b] - pos[a]
    vi = back_vel(a, par, pos, k); vo = fwd_vel(b, succ, pos, k)
    c = w_disp * float(d @ d)
    n = 0
    if vi is not None: c += float((d - vi) @ (d - vi)); n += 1
    if vo is not None: c += float((d - vo) @ (d - vo)); n += 1
    return c, n

def swap_repair(nodes, edges, margin=4.0, radius=10.0, dmax=12.0, k=2, w_disp=0.0, passes=2, min_terms=2, protect=None):
    edges = [dict(e) for e in edges]
    stats = {'swaps': 0}
    for _ in range(passes):
        succ, par, pos = build(nodes, edges)
        byt = defaultdict(list)
        for e in edges:
            s, d = int(e['source_id']), int(e['target_id'])
            if len(succ[s]) == 1: byt[int(nodes[s]['t'])].append((s, d))
        proposals = []
        for t, lst in byt.items():
            if len(lst) < 2: continue
            src = np.array([pos[s] for s, _ in lst]); tree = cKDTree(src)
            for i, j in tree.query_pairs(radius):
                (p, q), (r, s2) = lst[i], lst[j]
                if protect and (p in protect or r in protect or q in protect or s2 in protect): continue
                if np.linalg.norm(pos[s2] - pos[p]) > dmax or np.linalg.norm(pos[q] - pos[r]) > dmax: continue
                c1, n1 = jcost(p, q, succ, par, pos, k, w_disp); c2, n2 = jcost(r, s2, succ, par, pos, k, w_disp)
                c3, n3 = jcost(p, s2, succ, par, pos, k, w_disp); c4, n4 = jcost(r, q, succ, par, pos, k, w_disp)
                if min(n1 + n2, n3 + n4) < min_terms: continue
                gain = (c1 + c2) - (c3 + c4)
                if gain > margin: proposals.append((gain, p, q, r, s2))
        proposals.sort(reverse=True)
        used = set(); changes = []
        for gain, p, q, r, s2 in proposals:
            if used & {p, q, r, s2}: continue
            used |= {p, q, r, s2}; changes.append((p, q, r, s2))
        if not changes: break
        rm = set(); add = []
        for p, q, r, s2 in changes:
            rm |= {(p, q), (r, s2)}; add += [(p, s2), (r, q)]
        edges = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm] + [{'source_id': a, 'target_id': b, 'swap_repair': 1} for a, b in add]
        stats['swaps'] += len(changes)
    return edges, stats
