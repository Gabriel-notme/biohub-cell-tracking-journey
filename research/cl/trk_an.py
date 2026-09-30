"""Removal economics on B5 tracklets. Aggregate adj-J contribution of movie i = TP_i * m_i / W, m_i = 1 - 0.1 (Npred_i - Ntot_i)/Ntot_i.
Removing n nodes of movie i: +TP_i*0.1*n/Ntot_i ; losing k TP edges: -k*m_i. W approximated by #GT edges (TP+FN; FP small)."""
import glob, sys
import numpy as np
from collections import defaultdict
C = dict(len=0, t0=1, t1=2, st=3, en=4, z=5, y=6, x=7, sz=8, sy=9, sx=10, v=11, vmax=12, csz=13, ct0=14, ct1=15, ep=16, nm=17, tp=18)
D = defaultdict(list)
for f in sorted(glob.glob('/workspace/cl/trk/*.npz')):
    s, m = f.split('/')[-1][:-4].split('__')
    z = np.load(f); D[s].append((m, z['X'], float(z['n_total']), int(z['n_pred']), int(z['n_gt']), int(z['n_gt_edges'])))
print({s: len(v) for s, v in D.items()})


def evaluate(sel_fn, sets):
    gain = loss = W = 0.; nrem = ntp = 0
    for s in sets:
        for m, X, ntot, npred, ngt, nge in D[s]:
            TP = X[:, C['tp']].sum(); mi = 1 - 0.1 * (npred - ntot) / ntot
            sel = sel_fn(X)
            n = X[sel, C['len']].sum(); k = X[sel, C['tp']].sum()
            gain += TP * 0.1 * n / ntot; loss += k * mi; W += nge; nrem += n; ntp += k
    return (gain - loss) / W, gain / W, loss / W, nrem, ntp


ALL = ['t127a', 't127b', 'audit32', 'hold36', 'prev4']
# baseline stats
for s in ALL:
    if not D[s]: continue
    tot_nodes = sum(X[:, 0].sum() for _, X, *_ in D[s]); tot_tp = sum(X[:, C['tp']].sum() for _, X, *_ in D[s]); tot_nm = sum(X[:, C['nm']].sum() for _, X, *_ in D[s])
    mult = np.mean([1 - 0.1 * (npred - ntot) / ntot for _, X, ntot, npred, *_ in D[s]])
    print(s, 'movies', len(D[s]), 'nodes', int(tot_nodes), 'GT-matched nodes %.2f%%' % (100 * tot_nm / tot_nodes), 'TP', int(tot_tp), 'mean mult %.4f' % mult)
print('oracle remove tp==0 tracklets:', ['%s %.4f' % (s, evaluate(lambda X: X[:, C['tp']] == 0, [s])[0]) for s in ALL if D[s]])
print('oracle remove nm==0 tracklets:', ['%s %.4f' % (s, evaluate(lambda X: X[:, C['nm']] == 0, [s])[0]) for s in ALL if D[s]])
# by tracklet length
X = np.concatenate([X for s in ALL for _, X, *_ in D[s]])
L = X[:, 0]
for lo, hi in [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 99), (100, 100)]:
    k = (L >= lo) & (L <= hi)
    print('len %3d-%3d: tracklets %6d nodes %8d (%.1f%%)  GT-matched node rate %.3f%%  TP/node %.4f' % (lo, hi, k.sum(), L[k].sum(), 100 * L[k].sum() / L.sum(), 100 * X[k, C['nm']].sum() / max(1, L[k].sum()), X[k, C['tp']].sum() / max(1, L[k].sum())))
csz = X[:, C['csz']]
for lo, hi in [(1, 2), (3, 5), (6, 10), (11, 30), (31, 99), (100, 199), (200, 10 ** 9)]:
    k = (csz >= lo) & (csz <= hi)
    print('comp size %4d-%4d: nodes %8d (%.1f%%) GT-matched rate %.3f%% TP/node %.4f' % (lo, hi, L[k].sum(), 100 * L[k].sum() / L.sum(), 100 * X[k, C['nm']].sum() / max(1, L[k].sum()), X[k, C['tp']].sum() / max(1, L[k].sum())))
print('break-even TP/node ~ %.5f' % np.mean([0.1 * X_[:, C['tp']].sum() / ntot / (1 - 0.1 * (npred - ntot) / ntot) for s in ALL for _, X_, ntot, npred, *_ in D[s]]))
for th in [1, 2, 3, 5, 10, 20]:
    print('remove components with size <= %d:' % th, ['%s %+.4f (nodes %d tp %d)' % ((s,) + tuple(np.array(evaluate(lambda X: X[:, C['csz']] <= th, [s]))[[0, 3, 4]])) for s in ALL if D[s]])
