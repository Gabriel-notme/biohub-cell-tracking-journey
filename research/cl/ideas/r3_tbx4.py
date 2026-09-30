"""Temporal-boundary extension (critic A probe). A track that STARTS at frame T0+1..T0+K (or ENDS at T1-K..T1-1) and is at
least `minlen` nodes long is extended toward the movie boundary with DROPPED pre-ILP detections:
  mode 'edge': follow the pre-ILP (fullgraph) edge dropped_a(t-1) -> start(t) with prob >= pmin (END side: end(t) -> dropped_b(t+1));
  mode 'near': nearest dropped pre-ILP detection in the adjacent frame within r um of the current node.
The walk repeats until the boundary frame. Skips a detection that has an existing node within `dup` um in its frame
(the cell is already represented). Adds nodes + dt=1 edges only; never creates forks or merges. GT-free."""
WANTS_META = True
import sys
from collections import defaultdict
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])


def _rpos(v):
    return np.array([max(0, int(round(float(v[k])))) for k in 'zyx'], float) * S


def apply(nodes, edges, K=3, minlen=5, pmin=0.5, mode='edge', r=6.0, side='both', dup=3.5, maxstep=10 ** 6, join=False, **meta):
    sys.path.insert(0, '/workspace/p56stage')
    from edge_link import load_full
    from scipy.spatial import cKDTree
    fids, fT, fV, fE, fprob = load_full(meta['fullgeff'])
    fidx = {int(i): j for j, i in enumerate(fids.tolist())}
    fpar = defaultdict(list); fch = defaultdict(list)
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        fpar[int(b)].append((float(p), int(a))); fch[int(a)].append((float(p), int(b)))
    par = {}; succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); par[b] = a; succ[a].append(b)
    ts = [int(v['t']) for v in nodes.values()]; T0, T1 = min(ts), max(ts)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    trees = {t: cKDTree(np.stack([_rpos(nodes[n]) for n in ns])) for t, ns in byt.items()}
    drop_by_t = defaultdict(list)
    for j, (i, t) in enumerate(zip(fids.tolist(), fT.tolist())):
        if int(i) not in nodes: drop_by_t[int(t)].append(j)
    dtrees = {t: (ix, cKDTree(np.stack([_rpos({'z': fV[j][0], 'y': fV[j][1], 'x': fV[j][2]}) for j in ix]))) for t, ix in drop_by_t.items() if ix}

    def chain_len(n, down):
        L = 1; c = n
        while L < minlen:
            nx = succ.get(c, []) if down else ([par[c]] if c in par else [])
            if len(nx) != 1: break
            c = nx[0]; L += 1
        return L
    new_nodes = dict(nodes); new_edges = list(edges); used = set(); added = defaultdict(int)

    def free(j, t):
        if j in used or int(fids[j]) in new_nodes: return False
        if dup is not None and t in trees:
            p = _rpos({'z': fV[j][0], 'y': fV[j][1], 'x': fV[j][2]})
            if trees[t].query_ball_point(p, dup): return False
        return True

    def step(cur, t_next, backward):
        if mode == 'edge':
            cands = fpar.get(cur, []) if backward else fch.get(cur, [])
            cands = sorted([(p, x) for p, x in cands if p >= pmin and x in fidx and int(fT[fidx[x]]) == t_next], reverse=True)
            for p, x in cands:
                if free(fidx[x], t_next): return fidx[x]
            return None
        if t_next not in dtrees: return None
        ix, tr = dtrees[t_next]
        pc = _rpos(new_nodes[cur])
        for d, k in sorted(zip(*tr.query(pc, k=min(4, len(ix)))) if len(ix) > 1 else [tr.query(pc, k=1)]):
            if d > r: break
            if free(ix[int(k)], t_next): return ix[int(k)]
        return None
    seeds = []
    if side in ('both', 'start'):
        seeds += [(n, True) for n in nodes if n not in par and T0 < int(nodes[n]['t']) <= T0 + K and chain_len(n, True) >= minlen]
    if side in ('both', 'end'):
        seeds += [(n, False) for n in nodes if not succ.get(n) and T1 - K <= int(nodes[n]['t']) < T1 and chain_len(n, False) >= minlen]
    got_child = set(); got_par = set()
    for n, backward in seeds:
        if backward and n in got_par: continue
        if (not backward) and n in got_child: continue
        cur = n; nstep = 0
        while nstep < maxstep:
            nstep += 1
            t = int(new_nodes[cur]['t']); tn = t - 1 if backward else t + 1
            if tn < T0 or tn > T1: break
            j = step(cur, tn, backward)
            if j is None: break
            used.add(j); nid = int(fids[j])
            new_nodes[nid] = {'node_id': nid, 't': tn, 'z': float(fV[j][0]), 'y': float(fV[j][1]), 'x': float(fV[j][2]), 'tb_ext': 1}
            new_edges.append({'source_id': nid, 'target_id': cur, 'tb_ext': 1} if backward else {'source_id': cur, 'target_id': nid, 'tb_ext': 1})
            if backward: got_par.add(cur)
            else: got_child.add(cur)
            added['start' if backward else 'end'] += 1
            cur = nid
            if join:  # the dropped node's own pre-ILP neighbour on the far side is a kept free END (START): close the gap and stop
                far = fpar.get(cur, []) if backward else fch.get(cur, [])
                tf = tn - 1 if backward else tn + 1
                hit = None
                for p, x in sorted(far, reverse=True):
                    if p < pmin or x not in nodes or int(nodes[x]['t']) != tf: continue
                    if backward and not succ.get(x) and x not in got_child: hit = x; break
                    if (not backward) and x not in par and x not in got_par: hit = x; break
                if hit is not None:
                    if backward: new_edges.append({'source_id': hit, 'target_id': cur, 'tb_join': 1}); got_child.add(hit)
                    else: new_edges.append({'source_id': cur, 'target_id': hit, 'tb_join': 1}); got_par.add(hit)
                    added['join'] += 1
                    break
    return new_nodes, new_edges, {'tb_add_start': added['start'], 'tb_add_end': added['end'], 'tb_join': added['join'], 'tb_seeds': len(seeds)}
