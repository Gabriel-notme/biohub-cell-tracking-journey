"""Embryo-agnostic duplicate-track rule: remove whole fork-free isolated tracks whose nodes lie within 5 um of another track's node
in >= f of their frames (shadow/duplicate detections), optionally only if len <= L. No learned parameters.
Exact aggregated change per set and per embryo over all 199 movies (P8 track tables)."""
import pickle, sys
import numpy as np
sys.path.insert(0, '/workspace/cl')
from jprune import FEATS
iD = FEATS.index('dup5')
SETS = ['p8_t127a', 'p8_t127b', 'p8_audit32', 'p8_hold36', 'p8_prev4']
R = []
for t in SETS:
    for r in pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb')): r['set'] = t; R.append(r)


def agg(rs, f, L):
    n0 = d0 = n1 = d1 = 0.; nn = kk = 0
    for r in rs:
        m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
        sel = (r['X'][:, iD] >= f) & (r['len'] <= L) if len(r['len']) else np.zeros(0, bool)
        n = r['len'][sel].sum(); k = r['tp'][sel].sum(); fp = r['fp'][sel].sum(); nn += n; kk += k
        n0 += r['TP'] * m; d0 += w; n1 += (r['TP'] - k) * (m + 0.1 * n / r['ntot']); d1 += w - fp
    return n1 / d1 - n0 / d0, nn, kk


for f in [1.0, 0.9, 0.8, 0.6]:
    for L in [5, 10, 20, 100]:
        line = 'dup5>=%.1f len<=%3d:' % (f, L)
        for emb in ['44b6', '6bba']:
            g, n, k = agg([r for r in R if r['movie'].startswith(emb)], f, L); line += ' %s %+.5f (n%d tp%d) |' % (emb, g, n, k)
        g, n, k = agg(R, f, L); line += ' all199 %+.5f' % g
        print(line)
