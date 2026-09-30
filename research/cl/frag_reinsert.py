"""Temporarily re-insert short chains of dropped pre-ILP detections (not overlapping existing detections) so that the
learned linking steps can attach them to existing tracks; chains left unattached are removed again afterwards."""
from collections import defaultdict
import numpy as np

S = np.array([1.625, .40625, .40625])


def add(nodes, edges, full, max_len=5, clear_um=4.0):
    from scipy.spatial import cKDTree
    fids, fT, fV, fE, fprob = full
    idx = {int(i): j for j, i in enumerate(fids.tolist())}
    drop = np.array([int(i) not in nodes for i in fids.tolist()])
    fs = {}; fpar = {}
    for a, b in fE.tolist():
        if drop[idx[a]] and drop[idx[b]]: fs[a] = b; fpar[b] = a
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([[nodes[n][k] for k in 'zyx'] for n in ns]) * S) for t, ns in frames.items()}
    nn = dict(nodes); ne = list(edges)
    new = set(); nch = 0
    for j in np.where(drop)[0]:
        n0 = int(fids[j])
        if n0 in fpar: continue
        ch = [n0]
        while ch[-1] in fs and len(ch) <= max_len: ch.append(fs[ch[-1]])
        if len(ch) > max_len: continue
        ok = True
        for x in ch:
            t = int(fT[idx[x]]); p = fV[idx[x]] * S
            if t in trees and trees[t].query(p)[0] < clear_um: ok = False; break
        if not ok: continue
        if any(x in nn for x in ch): continue
        ids = []
        for x in ch:  # keep the pre-ILP id so that candidate-graph edge probabilities stay addressable
            k = idx[x]
            nn[x] = {'node_id': x, 't': int(fT[k]), 'z': float(fV[k][0]), 'y': float(fV[k][1]), 'x': float(fV[k][2]), 'reinserted': 1}
            ids.append(x); new.add(x)
        for a, b in zip(ids[:-1], ids[1:]): ne.append({'source_id': a, 'target_id': b, 'reinserted': 1})
        nch += 1
    return nn, ne, new, {'ri_chains': nch, 'ri_nodes': len(new)}


def cleanup_both(nodes, edges, new):
    """Keep a re-inserted chain only if it bridges two original nodes (original parent before its head and original
    child after its tail); remove all other re-inserted nodes."""
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    keep = set()
    for n in new:
        if n in par and par[n] in new: continue  # not a chain head
        ch = [n]
        while len(succ.get(ch[-1], [])) == 1 and succ[ch[-1]][0] in new: ch.append(succ[ch[-1]][0])
        head_ok = n in par and par[n] not in new
        tail_ok = any(c not in new for c in succ.get(ch[-1], []))
        if head_ok and tail_ok: keep.update(ch)
    drop = new - keep
    nn = {k: v for k, v in nodes.items() if k not in drop}
    ne = [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
    return nn, ne, {'ri_kept_nodes': len(keep), 'ri_removed_nodes': len(drop)}


def cleanup(nodes, edges, new):
    """Remove re-inserted nodes whose weakly connected component contains no original node."""
    adj = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); adj[a].append(b); adj[b].append(a)
    keep_new = set(); seen = set()
    for n in new:
        if n in seen: continue
        comp = []; st = [n]; seen.add(n)
        while st:
            x = st.pop(); comp.append(x)
            for y in adj[x]:
                if y not in seen: seen.add(y); st.append(y)
        if any(x not in new for x in comp): keep_new.update(x for x in comp if x in new)
    drop = new - keep_new
    nn = {k: v for k, v in nodes.items() if k not in drop}
    ne = [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
    return nn, ne, {'ri_kept_nodes': len(keep_new), 'ri_removed_nodes': len(drop)}
