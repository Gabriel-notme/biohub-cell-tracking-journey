"""P17 deletion-type structure clean-up on the final P-stage graph (GT-free; removes implausible structure only).
  cutdup     : remove surviving B5 gap bridges (gap_closed / gap2_recovered edges) whose synthetic node lies within 3.2 um of another
               node of its frame (the cell is already represented), dropping the whole synthetic chain.
  forkfrag   : remove weakly connected components with < 6 nodes and no fork that contain a former fork daughter of the B5 reference
               graph that now has no parent (fragments B5's own short-track filter never re-checked after later stages cut the fork).
  shortbranch: a fork whose daughter branch ends within 2 frames (<= 3 nodes, no further fork) is not a division: delete that branch."""
import json
from collections import defaultdict
import numpy as np

S = np.array([1.625, .40625, .40625])


def _pos(nodes):
    return {n: np.array([float(v['z']), float(v['y']), float(v['x'])]) * S for n, v in nodes.items()}


def cutdup(nodes, edges, rad=3.2):
    from scipy.spatial import cKDTree
    succ = defaultdict(list); par = {}; fl = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
        fl[(a, b)] = 'gc' if 'gap_closed' in e else ('g2' if 'gap2_recovered' in e else 'o')
    pos = _pos(nodes); ids_by_t = defaultdict(list)
    for n, v in nodes.items(): ids_by_t[int(v['t'])].append(n)
    trees = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in ids_by_t.items()}
    syn = [n for n, v in nodes.items() if n in par and fl.get((par[n], n)) in ('gc', 'g2') and len(succ.get(n, [])) == 1
           and fl.get((n, succ[n][0])) in ('gc', 'g2') and ('gap_synthetic' in v or fl.get((par[n], n)) == 'g2')]
    syns = set(syn); rm = set()
    for n in syn:
        ns, tr = trees[int(nodes[n]['t'])]
        if any(ns[k] != n for k in tr.query_ball_point(pos[n], rad)):
            chain = [n]; c = n
            while c in par and fl.get((par[c], c)) in ('gc', 'g2') and par[c] in syns: c = par[c]; chain.append(c)
            c = n
            while len(succ.get(c, [])) == 1 and fl.get((c, succ[c][0])) in ('gc', 'g2') and succ[c][0] in syns: c = succ[c][0]; chain.append(c)
            rm.update(chain)
    if not rm: return nodes, edges, 0
    return ({k: v for k, v in nodes.items() if k not in rm},
            [e for e in edges if int(e['source_id']) not in rm and int(e['target_id']) not in rm], len(rm))


def forkfrag(nodes, edges, refp, minlen=6):
    d = json.loads(open(refp).read()); ro = defaultdict(list)
    for e in d['edges']: ro[int(e['source_id'])].append(int(e['target_id']))
    D = {c for s, cs in ro.items() if len(cs) >= 2 for c in cs}
    par = {}; out = defaultdict(list); adj = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); par[b] = a; out[a].append(b); adj[a].append(b); adj[b].append(a)
    seen = set(); rm = set()
    for n in nodes:
        if n in seen: continue
        comp = []; stack = [n]; seen.add(n)
        while stack:
            u = stack.pop(); comp.append(u)
            for v in adj.get(u, []):
                if v not in seen: seen.add(v); stack.append(v)
        if len(comp) >= minlen or any(len(out.get(u, [])) >= 2 for u in comp): continue
        if not any(u in D and u not in par for u in comp): continue
        rm.update(comp)
    if not rm: return nodes, edges, 0
    return ({k: v for k, v in nodes.items() if k not in rm},
            [e for e in edges if int(e['source_id']) not in rm and int(e['target_id']) not in rm], len(rm))


def short_branch(nodes, edges, maxk=2):
    out = defaultdict(list)
    for e in edges: out[int(e['source_id'])].append(int(e['target_id']))
    drop = set(); nf = 0
    for p, ch in list(out.items()):
        if len(ch) != 2: continue
        for c in ch:
            br = [c]; n = c; ok = True
            while True:
                nx = out.get(n, [])
                if len(nx) == 0: break
                if len(nx) == 2 or len(br) > maxk: ok = False; break
                n = nx[0]; br.append(n)
            if ok and len(br) <= maxk + 1:
                drop.update(br); nf += 1; break
    if not drop: return nodes, edges, 0, 0
    return ({k: v for k, v in nodes.items() if k not in drop},
            [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop], nf, len(drop))


def apply(nodes, edges, refp, cutdup_on=1, forkfrag_on=1, shortbranch_on=1):
    st = {}
    if cutdup_on:
        nodes, edges, k = cutdup(nodes, edges); st['p17_cd'] = k
    if forkfrag_on:
        nodes, edges, k = forkfrag(nodes, edges, refp); st['p17_ff'] = k
    if shortbranch_on:
        nodes, edges, f, k = short_branch(nodes, edges); st['p17_sbf'] = f; st['p17_sb'] = k
    return nodes, edges, st
