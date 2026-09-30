"""Post-hoc long-range free-end linking on final P13 graphs (no model, no GT).
edge_link caps gap-1 candidates at RMAX[1]=14 um, and relink skips pure free-end pairs ('handled by edge_link'), so pre-ILP
candidate edges longer than 14 um between a track end (no child) at t and a track start (no parent) at t+1 are never considered.
This rule adds them greedily by pre-ILP edge_prob (fe >= fe_min), one link per end/start, only if longer than min_um.
kwargs: fe_min (0.5), min_um (14.0), max_um (1e9)."""
import numpy as np
WANTS_META = True
S = np.array([1.625, .40625, .40625])


def apply(nodes, edges, name=None, set=None, fullgeff=None, zarr=None, fe_min=0.5, min_um=14.0, max_um=1e9):
    import sys
    sys.path.insert(0, '/workspace/p56stage')
    from edge_link import load_full
    fids, fT, fV, fE, fprob = load_full(fullgeff)
    has_child = {int(e['source_id']) for e in edges}; has_par = {int(e['target_id']) for e in edges}
    cand = []
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        a, b = int(a), int(b)
        if p < fe_min or a not in nodes or b not in nodes or a in has_child or b in has_par: continue
        if int(nodes[b]['t']) != int(nodes[a]['t']) + 1: continue
        d = float(np.linalg.norm((np.array([nodes[a][k] for k in 'zyx'], float) - np.array([nodes[b][k] for k in 'zyx'], float)) * S))
        if min_um < d <= max_um: cand.append((p, a, b, d))
    add = []
    for p, a, b, d in sorted(cand, reverse=True):
        if a in has_child or b in has_par: continue
        has_child.add(a); has_par.add(b); add.append({'source_id': a, 'target_id': b, 'long_link': round(p, 4)})
    return nodes, list(edges) + add, {'ll_cands': len(cand), 'll_added': len(add)}
