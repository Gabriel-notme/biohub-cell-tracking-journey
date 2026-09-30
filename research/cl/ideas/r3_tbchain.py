"""Boundary-truncated chain reinsertion (critic A probe).
B5 removes tracks shorter than 6 nodes. A real cell whose first (last) frames form a short pre-ILP chain that the ILP did not join to
its main track is truncated by the movie boundary on one side, so it is removed; in the interior the same fragment has two open
sides. Rule: take chains of DROPPED pre-ILP detections joined by pre-ILP edges (prob >= pmin, each node in/out-degree <= 1 inside the
dropped subgraph), length >= Lmin, that contain the first frame (T0) or the last frame (T1) (side='boundary'; side='interior' is
the control: chains touching neither). Skip a chain if any node has a kept node within `dup` um in its frame. Insert the chain.
link: 'none' | 'edge' (inner end -> kept START via a pre-ILP edge; mirror for end side) | 'near' (nearest kept START/END in the
adjacent frame within r um). Adds nodes + dt=1 edges, never forks or merges. GT-free."""
WANTS_META = True
import sys
from collections import defaultdict
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])


def _r(v):
    return np.array([max(0, int(round(float(x)))) for x in v], float)


def apply(nodes, edges, side='boundary', Lmin=2, pmin=0.5, link='edge', r=6.0, dup=3.5, **meta):
    sys.path.insert(0, '/workspace/p56stage')
    from edge_link import load_full
    from scipy.spatial import cKDTree
    fids, fT, fV, fE, fprob = load_full(meta['fullgeff'])
    fidx = {int(i): j for j, i in enumerate(fids.tolist())}
    par = {}; succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); par[b] = a; succ[a].append(b)
    ts = [int(v['t']) for v in nodes.values()]; T0, T1 = min(ts), max(ts)
    drop = {int(i) for i in fids.tolist() if int(i) not in nodes}
    dout = defaultdict(list); din = defaultdict(list); kin = defaultdict(list); kout = defaultdict(list)
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        a, b = int(a), int(b)
        if a not in fidx or b not in fidx or int(fT[fidx[b]]) != int(fT[fidx[a]]) + 1: continue
        if a in drop and b in drop and p >= pmin: dout[a].append(b); din[b].append(a)
        elif a in drop and b in nodes: kout[a].append((p, b))   # dropped -> kept
        elif a in nodes and b in drop: kin[b].append((p, a))    # kept -> dropped
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    kpos = {n: _r([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    ktree = {t: (cKDTree(np.stack([kpos[n] for n in ns])), ns) for t, ns in byt.items()}
    # chains: start at dropped nodes with no dropped parent
    chains = []
    for a in drop:
        if din.get(a): continue
        ch = [a]; ok = True
        while True:
            nx = dout.get(ch[-1], [])
            if len(nx) == 0: break
            if len(nx) > 1 or len(din.get(nx[0], [])) > 1: ok = False; break
            ch.append(nx[0])
        if ok and len(ch) >= Lmin: chains.append(ch)
    new_nodes = dict(nodes); new_edges = list(edges); st = defaultdict(int)
    kstart_used = set(); kend_used = set()
    for ch in chains:
        t_first = int(fT[fidx[ch[0]]]); t_last = int(fT[fidx[ch[-1]]])
        touches0 = t_first == T0; touches1 = t_last == T1
        if side == 'boundary' and not (touches0 or touches1): continue
        if side == 'start' and not touches0: continue
        if side == 'end' and not touches1: continue
        if side == 'interior' and (touches0 or touches1): continue
        bad = False
        for x in ch:
            j = fidx[x]; t = int(fT[j])
            if t in ktree and ktree[t][0].query_ball_point(_r(fV[j]) * S, dup): bad = True; break
        if bad: st['skip_dup'] += 1; continue
        for x in ch:
            j = fidx[x]
            new_nodes[x] = {'node_id': x, 't': int(fT[j]), 'z': float(fV[j][0]), 'y': float(fV[j][1]), 'x': float(fV[j][2]), 'tb_chain': 1}
        for a, b in zip(ch[:-1], ch[1:]): new_edges.append({'source_id': a, 'target_id': b, 'tb_chain': 1})
        st['chains'] += 1; st['nodes'] += len(ch)
        # link the open (inner) side(s)
        for inner_is_last in ([True] if touches0 and not touches1 else [False] if touches1 and not touches0 else [True, False]):
            x = ch[-1] if inner_is_last else ch[0]
            tx = int(fT[fidx[x]]); tn = tx + 1 if inner_is_last else tx - 1
            if tn < T0 or tn > T1: continue
            tgt = None
            if link == 'edge':
                cands = sorted(kout.get(x, []) if inner_is_last else kin.get(x, []), reverse=True)
                for p, k in cands:
                    if p < pmin: continue
                    if inner_is_last and k not in par and k not in kstart_used: tgt = k; break
                    if (not inner_is_last) and not succ.get(k) and k not in kend_used: tgt = k; break
            elif link == 'near' and tn in ktree:
                tr, ns = ktree[tn]
                px = _r(fV[fidx[x]]) * S
                for q in tr.query_ball_point(px, r):
                    k = ns[q]
                    if inner_is_last and k not in par and k not in kstart_used:
                        if tgt is None or np.linalg.norm(kpos[k] - px) < np.linalg.norm(kpos[tgt] - px): tgt = k
                    if (not inner_is_last) and not succ.get(k) and k not in kend_used:
                        if tgt is None or np.linalg.norm(kpos[k] - px) < np.linalg.norm(kpos[tgt] - px): tgt = k
            if tgt is not None:
                if inner_is_last: new_edges.append({'source_id': x, 'target_id': tgt, 'tb_chain_link': 1}); kstart_used.add(tgt)
                else: new_edges.append({'source_id': tgt, 'target_id': x, 'tb_chain_link': 1}); kend_used.add(tgt)
                st['links'] += 1
    return new_nodes, new_edges, {'tbc_' + k: v for k, v in st.items()}
