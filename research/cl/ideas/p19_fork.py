"""p19_fork: deletion-only clean-up of implausible forks on the final P17 graph (GT-free, constants from B5 / physics).
  mode 'gapcut'  : fork p->(a,b) where one child edge is a B5 gap bridge (gap_closed / gap2_recovered, synthetic daughter): B5's gap
                   closer only bridges a track END, so the second child was attached later; the interpolated daughter has no detection
                   at t+1 -> remove the synthetic bridge chain (edge p->c and the synthetic nodes).
  mode 'othercut': same forks, but cut the other (non-bridge, later-added) child edge instead.
  mode 't0cut'   : fork at the first frame of the movie (no history; 2x excess fork rate there): cut the edge to the child whose
                   branch is shorter (frames to its end / next fork; tie -> larger displacement from p).
  mode 'dupcut'  : fork whose sisters come within 3.2 um (B5 GAP_CLOSE_REUSE_UM) of each other in any of the first 3 daughter frames,
                   or whose daughter lies within 3.2 um of a non-sister node at t+1 (one cell detected twice): cut the edge to the
                   shorter-branch child (for the neighbour case: that daughter).
After any cut, the component that now contains the cut child is deleted if it has < 6 nodes (B5 OUTPUT_MIN_TRACK_LEN) and no fork."""
from collections import defaultdict
import numpy as np
S = np.array([1.625, .40625, .40625])
GAP = ('gap_closed', 'gap2_recovered')
REUSE = 3.2
MINLEN = 6


def _isgap(e):
    return any(k in e for k in GAP)


def apply(nodes, edges, mode='gapcut', minlen=MINLEN, **kw):
    out = defaultdict(list); par = {}; ea = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a; ea[(a, b)] = e
    pos = {n: np.array([float(v['z']), float(v['y']), float(v['x'])]) * S for n, v in nodes.items()}
    T0 = min(int(v['t']) for v in nodes.values())

    def blen(x):
        k = 1
        while len(out.get(x, [])) == 1 and k < 200: x = out[x][0]; k += 1
        return k

    def syn(n):
        return 'gap_synthetic' in nodes[n] or (n in par and 'gap2_recovered' in ea[(par[n], n)])

    cut = set(); rmn = set(); st = defaultdict(int)
    forks = [p for p, ch in out.items() if len(ch) == 2]
    if mode == 'dupcut':
        from scipy.spatial import cKDTree
        byt = defaultdict(list)
        for n, v in nodes.items(): byt[int(v['t'])].append(n)
        trees = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in byt.items()}
    for p in sorted(forks, key=lambda n: (int(nodes[n]['t']), n)):
        a, b = out[p]
        tgt = None
        if mode in ('gapcut', 'othercut'):
            g = [c for c in (a, b) if _isgap(ea[(p, c)]) or 'gap_synthetic' in nodes[c]]
            if len(g) != 1: continue
            c = g[0]; o = b if c == a else a
            st['forks'] += 1
            if mode == 'othercut':
                tgt = o
            else:
                chain = [c]; x = c
                while syn(x) and len(out.get(x, [])) == 1 and _isgap(ea[(x, out[x][0])]) and syn(out[x][0]):
                    x = out[x][0]; chain.append(x)
                if not syn(c): chain = []
                cut.add((p, c)); rmn.update(chain)
                if chain and len(out.get(chain[-1], [])) == 1: cut.add((chain[-1], out[chain[-1]][0]))
                continue
        elif mode == 't0cut':
            if int(nodes[p]['t']) != T0: continue
            la, lb = blen(a), blen(b)
            if la != lb: tgt = a if la < lb else b
            else: tgt = a if np.linalg.norm(pos[a] - pos[p]) >= np.linalg.norm(pos[b] - pos[p]) else b
            st['forks'] += 1
        elif mode == 'dupcut':
            ba, bb = [a], [b]
            for br in (ba, bb):
                while len(br) < 3 and len(out.get(br[-1], [])) == 1: br.append(out[br[-1]][0])
            dup = any(np.linalg.norm(pos[x] - pos[y]) < REUSE for x, y in zip(ba, bb))
            if dup:
                la, lb = blen(a), blen(b); tgt = a if la < lb else (b if lb < la else (a if np.linalg.norm(pos[a] - pos[p]) >= np.linalg.norm(pos[b] - pos[p]) else b))
                st['sisdup'] += 1
            else:
                t1 = int(nodes[p]['t']) + 1; ns, tr = trees[t1]
                nb = [c for c in (a, b) if any(ns[k] not in (a, b) for k in tr.query_ball_point(pos[c], REUSE))]
                if len(nb) == 1: tgt = nb[0]; st['nbdup'] += 1
            if tgt is None: continue
        if tgt is not None: cut.add((p, tgt))
    if not cut and not rmn: return nodes, edges, dict(st)
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in cut and int(e['source_id']) not in rmn and int(e['target_id']) not in rmn]
    nn = {k: v for k, v in nodes.items() if k not in rmn}
    # fragment check on the components of the cut children / chain successors
    out2 = defaultdict(list); adj = defaultdict(list)
    for e in ne:
        x, y = int(e['source_id']), int(e['target_id']); out2[x].append(y); adj[x].append(y); adj[y].append(x)
    seeds = [c for (_, c) in cut if c in nn]
    frag = set(); seen = set()
    for s0 in seeds:
        if s0 in seen: continue
        comp = []; stack = [s0]; seen.add(s0)
        while stack:
            u = stack.pop(); comp.append(u)
            for v in adj.get(u, []):
                if v not in seen: seen.add(v); stack.append(v)
        if len(comp) < minlen and not any(len(out2.get(u, [])) >= 2 for u in comp): frag.update(comp)
    st['cut_edges'] = len(cut); st['syn_removed'] = len(rmn); st['frag_removed'] = len(frag)
    if frag:
        nn = {k: v for k, v in nn.items() if k not in frag}
        ne = [e for e in ne if int(e['source_id']) not in frag and int(e['target_id']) not in frag]
    return nn, ne, dict(st)
