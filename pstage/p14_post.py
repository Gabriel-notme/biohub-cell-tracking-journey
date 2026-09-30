"""P14 post-steps, applied after post_prune on the final P-stage graph. No learned parameters, no model.

term_trim: a track END node a (has a parent, no child) lying within r um (rounded submission coordinates) of a same-frame node m that
  continues (has a child) is a duplicate detection: it can supply at most one edge but can steal the 7 um match from m. Delete a.
  If m is itself a START with one child c, keep a and hand it m's continuation instead (a -> c, drop m). Fork daughters are never
  touched; the trimmed track must have >= minlen nodes; repeated up to iters times so a duplicated tail is removed completely.
long_link: edge_link only generates gap-1 candidates within 14 um, so pre-ILP candidate edges longer than that between a track end
  (no child) at t and a track start (no parent) at t+1 are never considered. Add them greedily by pre-ILP edge_prob (>= fe_min),
  each end/start used once. No nodes and no forks are created.
"""
from collections import defaultdict
import numpy as np

S = np.array([1.625, 0.40625, 0.40625])


def term_trim(nodes, edges, r=3.5, minlen=3, iters=5, join='keep_end'):
    from scipy.spatial import cKDTree
    nodes = dict(nodes); edges = list(edges)
    total = 0
    for it in range(iters):
        out = defaultdict(list); par = {}
        for e in edges:
            a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
        pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
        byt = defaultdict(list)
        for n, v in nodes.items(): byt[int(v['t'])].append(n)

        def tlen_back(n):
            L = 1
            while n in par: n = par[n]; L += 1
            return L
        drop = set(); add = []
        for t, ns in byt.items():
            if len(ns) < 2: continue
            tr = cKDTree(np.stack([pos[n] for n in ns]))
            for n in ns:
                if out[n] or n not in par: continue  # only track ends
                if len(out.get(par[n], [])) == 2: continue  # fork daughter
                if tlen_back(n) < minlen: continue
                for j in tr.query_ball_point(pos[n], r):
                    m = ns[j]
                    if m == n or m in drop or not out[m]: continue
                    if join and m not in par and len(out[m]) == 1 and it == 0:
                        if join == 'keep_end': drop.add(m); add.append((n, out[m][0]))
                        else: drop.add(n); add.append((par[n], m))
                    else:
                        drop.add(n)
                    break
        if not drop: break
        total += len(drop)
        for n in drop: nodes.pop(n, None)
        edges = [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
        edges += [{'source_id': a, 'target_id': b, 'dup_join': 1} for a, b in add if a not in drop and b not in drop]
    return nodes, edges, {'trim': total}


def long_link(nodes, edges, fullgeff, fe_min=0.5, min_um=14.0, max_um=1e9):
    from edge_link import load_full
    fids, fT, fV, fE, fprob = load_full(fullgeff)
    has_child = {int(e['source_id']) for e in edges}; has_par = {int(e['target_id']) for e in edges}
    cand = []
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        a, b = int(a), int(b)
        if p < fe_min or a not in nodes or b not in nodes or a in has_child or b in has_par: continue
        if int(nodes[b]['t']) != int(nodes[a]['t']) + 1: continue
        d = float(np.linalg.norm((np.array([nodes[a][k] for k in 'zyx'], float) - np.array([nodes[b][k] for k in 'zyx'], float)) * S))
        if min_um < d <= max_um: cand.append((p, a, b))
    add = []
    for p, a, b in sorted(cand, reverse=True):
        if a in has_child or b in has_par: continue
        has_child.add(a); has_par.add(b); add.append({'source_id': a, 'target_id': b, 'long_link': round(p, 4)})
    return nodes, list(edges) + add, {'ll_cands': len(cand), 'll_added': len(add)}
