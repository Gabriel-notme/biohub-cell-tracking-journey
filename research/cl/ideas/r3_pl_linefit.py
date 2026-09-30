"""Critic-B probe (B5 output-filter parameter): linefit temporal smoothing (weight 0.8, window 2) moves raw detections
(pre-ILP fullgraph coords) to the reference-graph coords before the centroid stage. Undo/extend it by a fraction a of that
displacement, keeping the centroid correction: new = cur + a * (raw - ref). a=0 reproduces P14; a=1 ~ no linefit; a<0 = stronger.
Only nodes whose id exists in both the fullgraph and the reference graph in the same frame and |raw-ref| <= 3 um are moved.
Same 2 um collision rejection as B5's centroid stage."""
import sys, json
import builtins
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/p56stage')
WANTS_META = True
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}
S = np.array([1.625, .40625, .40625]); SHAPE = np.array([64, 256, 256])


def apply(nodes, edges, name=None, set=None, fullgeff=None, zarr=None, a=0.0, minsep=2.0):
    from scipy.spatial import cKDTree
    from edge_link import load_full
    ref = json.load(open(REF[set] + '/' + name + '.json'))['nodes']
    fids, fT, fV, fE, fprob = load_full(fullgeff)
    raw = {int(i): (int(t), v) for i, t, v in zip(fids.tolist(), fT.tolist(), fV)}
    new = {k: dict(v) for k, v in nodes.items()}; changed = builtins.set(); skipped_far = 0
    for k, v in nodes.items():
        rv = ref.get(str(k)); fw = raw.get(int(k))
        if rv is None or fw is None or fw[0] != int(v['t']) or int(rv['t']) != int(v['t']): continue
        disp = np.asarray(fw[1], float) - np.array([rv[c] for c in 'zyx'], float)
        if not np.any(disp): continue
        if np.linalg.norm(disp * S) > 3.0: skipped_far += 1; continue
        q = np.array([v[c] for c in 'zyx'], float) + a * disp
        if np.any(q < 0) or np.any(np.rint(q) >= SHAPE): continue
        new[k].update(dict(zip('zyx', map(float, q)))); changed.add(k)
    frames = defaultdict(list)
    for k, v in nodes.items(): frames[int(v['t'])].append(k)
    rej_total = 0
    for ns in frames.values():
        if len(ns) < 2: continue
        old = np.array([[nodes[n][c] for c in 'zyx'] for n in ns]) * S
        for _ in range(3):
            pp = np.array([[new[n][c] for c in 'zyx'] for n in ns]) * S
            rej = builtins.set()
            for i, j in cKDTree(pp).query_pairs(minsep):
                if np.linalg.norm(pp[i] - pp[j]) + 1e-5 < min(minsep, np.linalg.norm(old[i] - old[j])):
                    rej.update(n for n in (ns[i], ns[j]) if n in changed)
            if not rej: break
            for n in rej: new[n] = dict(nodes[n]); changed.discard(n)
            rej_total += len(rej)
    return new, edges, {'lf_moved': len(changed), 'lf_rejected': rej_total, 'lf_far': skipped_far}
