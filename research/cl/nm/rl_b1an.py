"""b1 edge-head discrimination on rl_cands rows: in-sample (audit32/t127) vs b1-unseen (hold36/prev4) movies, per embryo and type."""
import os, glob, pickle, sys
import numpy as np
sys.path.insert(0, '/workspace/cl/nm')
from rl_cands import FEATS
D = []
for f in sorted(glob.glob('/workspace/cl/nm/rl_cands/*.pkl')):
    s, m = os.path.basename(f)[:-4].split('__'); z = pickle.load(open(f, 'rb')); e = pickle.load(open('/workspace/cl/nm/rl_b1f/' + os.path.basename(f), 'rb'))
    L = z['L']; y = ((L['tp_sd'] == 1) & (L['tp_sc'] == 0) & (L['tp_qd'] == 0)).astype(int)
    D.append((s, m[:4], z['X'], e['X'], y, L))
B = pickle.load(open('/workspace/cl/nm/rl_b1f/' + os.path.basename(f), 'rb'))['names']


def auc(p, y):
    o = np.argsort(p, kind='mergesort'); r = np.empty(len(p)); r[o] = np.arange(len(p)); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


def prec_at(p, y, ths):
    return ' '.join('p>=%.2f:%d/%d' % (t, int(y[p >= t].sum()), int((p >= t).sum())) for t in ths)


for grp, sets in [('clean(b1-unseen)', ('hold36', 'prev4')), ('in-sample', ('audit32', 't127a', 't127b'))]:
    for emb in ['44b6', '6bba']:
        sel = [d for d in D if d[0] in sets and d[1] == emb]
        X = np.concatenate([d[2] for d in sel]); Bx = np.concatenate([d[3] for d in sel]); y = np.concatenate([d[4] for d in sel])
        typ = X[:, FEATS.index('typ')]
        print('== %s %s rows %d pos %d' % (grp, emb, len(y), y.sum()))
        for k in ['b_sd', 'b_dsc', 'b_dqd', 'b_swap', 'cos_sd']:
            print('   AUC %-7s %.3f' % (k, auc(Bx[:, B.index(k)], y)), end='')
        print()
        for t in range(4):
            k = typ == t
            if y[k].sum() == 0: continue
            print('   typ %d rows %6d pos %4d AUC b_sd %.3f | %s' % (t, k.sum(), y[k].sum(), auc(Bx[k, B.index('b_sd')], y[k]), prec_at(Bx[k, B.index('b_sd')], y[k], [0.5, 0.8, 0.9, 0.95, 0.98])))
