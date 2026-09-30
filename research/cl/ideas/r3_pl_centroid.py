"""Critic-B probe (model-output angle): B5 'centroid' lineage stage blends each node toward the centroid_v1 CNN prediction with
centroid_blend=0.5 (displacement d clipped to 3 um, skipped if predicted error > 2.5 um, collisions < 2 um rejected).
The B5 reference graph holds the exact pre-centroid coordinates (max |lineage - ref| = 1.5 um = 0.5 * 3 um), so
d = 2 * (cur - ref) and any blend b is: new = ref + 2b * (cur - ref). b=0.5 reproduces P14. Nodes absent from the reference graph
(P-stage gap2 inserts, B5 recovered nodes) are left unchanged. The same min-separation collision rejection as B5 is re-applied."""
import json
import builtins
from collections import defaultdict
import numpy as np
WANTS_META = True
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}
S = np.array([1.625, .40625, .40625]); SHAPE = np.array([64, 256, 256])


def apply(nodes, edges, name=None, set=None, fullgeff=None, zarr=None, b=0.5, minsep=2.0, zonly=None):
    from scipy.spatial import cKDTree
    ref = json.load(open(REF[set] + '/' + name + '.json'))['nodes']
    f = 2.0 * b
    new = {k: dict(v) for k, v in nodes.items()}; changed = builtins.set()
    for k, v in nodes.items():
        rv = ref.get(str(k))
        if rv is None or int(rv['t']) != int(v['t']): continue
        c = np.array([v[a] for a in 'zyx'], float); r = np.array([rv[a] for a in 'zyx'], float)
        if np.allclose(c, r): continue
        q = r + f * (c - r)
        if zonly == 'z': q = np.array([q[0], c[1], c[2]])
        elif zonly == 'xy': q = np.array([c[0], q[1], q[2]])
        if np.any(q < 0) or np.any(np.rint(q) >= SHAPE): continue
        new[k].update(dict(zip('zyx', map(float, q)))); changed.add(k)
    frames = defaultdict(list)
    for k, v in nodes.items(): frames[int(v['t'])].append(k)
    rej_total = 0
    for ns in frames.values():
        if len(ns) < 2: continue
        old = np.array([[nodes[n][a] for a in 'zyx'] for n in ns]) * S
        for _ in range(3):
            pp = np.array([[new[n][a] for a in 'zyx'] for n in ns]) * S
            rej = builtins.set()
            for i, j in cKDTree(pp).query_pairs(minsep):
                if np.linalg.norm(pp[i] - pp[j]) + 1e-5 < min(minsep, np.linalg.norm(old[i] - old[j])):
                    rej.update(n for n in (ns[i], ns[j]) if n in changed)
            if not rej: break
            for n in rej: new[n] = dict(nodes[n]); changed.discard(n)
            rej_total += len(rej)
    return new, edges, {'cen_moved': len(changed), 'cen_rejected': rej_total}
