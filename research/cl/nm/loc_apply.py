"""Apply LOEO regressor offsets (loc_train.py predictions on the held-out embryo) to P15 node coordinates.
kw: tw (1|3: which training run), blend, gate (apply only if |delta| >= gate um), cap (max shift um).
Predictions exist for every intended node in 44b6 and for all tail/U + 35% of the other intended nodes in 6bba."""
import glob, pickle, os
import numpy as np
WANTS_META = True
S = np.array([1.625, .40625, .40625])
_C = {}


def _load(tw):
    if tw in _C: return _C[tw]
    D = {}
    for emb in ['44b6', '6bba']:
        P = np.load('/workspace/cl/nm/loc_pred_%s_b5_%g.npy' % (emb, tw)); k = 0
        for f in sorted(glob.glob('/workspace/cl/nm/loc_ds/%s_*.npz' % emb)):
            ids = np.load(f)['ids']; D[os.path.basename(f)[:-4]] = dict(zip(ids.tolist(), P[k:k + len(ids)])); k += len(ids)
        assert k == len(P)
    _C[tw] = D; return D


def apply(nodes, edges, tw=3, blend=1.0, gate=0.0, cap=99., name=None, **kw):
    D = _load(tw).get(name, {})
    out = {n: dict(v) for n, v in nodes.items()}; moved = 0
    for n, d in D.items():
        L = float(np.linalg.norm(d))
        if L < gate or n not in out: continue
        d = d * min(1., cap / max(L, 1e-9)) * blend
        v = out[n]
        for k, a in zip('zyx', range(3)): v[k] = float(v[k] + d[a] / S[a])
        moved += 1
    return out, edges, {'moved': moved}
