"""Can we predict which tracklets carry GT (annotated-lineage) edges? Poisson GBM on tracklet features -> expected TP count.
Remove tracklet if E[TP] < lam * len * breakeven_i. lam chosen on train OOF (movie-grouped CV), evaluated on hold36/prev4."""
import glob
import numpy as np
from collections import defaultdict
import lightgbm as lgb
C = dict(len=0, t0=1, t1=2, st=3, en=4, z=5, y=6, x=7, sz=8, sy=9, sx=10, v=11, vmax=12, csz=13, ct0=14, ct1=15, ep=16, nm=17, tp=18)
R = []
for f in sorted(glob.glob('/workspace/cl/trk/*.npz')):
    s, m = f.split('/')[-1][:-4].split('__')
    z = np.load(f); X = z['X']; ntot = float(z['n_total']); npred = int(z['n_pred']); nge = int(z['n_gt_edges'])
    TP = X[:, C['tp']].sum(); mi = 1 - 0.1 * (npred - ntot) / ntot
    be = 0.1 * TP / ntot / mi  # break-even TP per node
    # per-movie relative features
    zr = (X[:, C['z']] - X[:, C['z']].min()) / max(1e-6, np.ptp(X[:, C['z']]))
    yc = X[:, C['y']] - 128; xc = X[:, C['x']] - 128; rr = np.sqrt(yc ** 2 + xc ** 2)
    lrank = np.argsort(np.argsort(X[:, 0])) / len(X)
    F = np.column_stack([X[:, :17], zr, rr, lrank, (X[:, C['ct0']] == 0), (X[:, C['ct1']] >= 99), np.full(len(X), npred / ntot), np.full(len(X), len(X))])
    R.append(dict(s=s, m=m, X=X, F=F, y=X[:, C['tp']], be=be, TP=TP, mi=mi, ntot=ntot, W=nge))
print({s: sum(r['s'] == s for r in R) for s in set(r['s'] for r in R)})
TR = [r for r in R if r['s'] in ('t127a', 't127b', 'audit32')]
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=32)
rng = np.random.default_rng(0); idx = np.arange(len(TR)); rng.shuffle(idx); fold = {int(i): k % 5 for k, i in enumerate(idx)}
for k in range(5):
    tr = [r for i, r in enumerate(TR) if fold[i] != k]; va = [r for i, r in enumerate(TR) if fold[i] == k]
    b = lgb.train(params, lgb.Dataset(np.concatenate([r['F'] for r in tr]), np.concatenate([r['y'] for r in tr])), 400)
    for r in va: r['oof'] = b.predict(r['F'])
full = lgb.train(params, lgb.Dataset(np.concatenate([r['F'] for r in TR]), np.concatenate([r['y'] for r in TR])), 400)
for r in R:
    if r['s'] not in ('t127a', 't127b', 'audit32'): r['oof'] = full.predict(r['F'])


def net(rs, lam):
    g = l = W = 0.
    for r in rs:
        sel = r['oof'] < lam * r['X'][:, 0] * r['be']
        n = r['X'][sel, 0].sum(); k = r['y'][sel].sum()
        g += r['TP'] * 0.1 * n / r['ntot']; l += k * r['mi']; W += r['W']
    return (g - l) / W, g / W, l / W


def auc(p, y):
    o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(len(p)); n1 = y.sum(); n0 = len(y) - n1
    return (rk[y == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


for s in ['train', 'hold36', 'prev4']:
    rs = TR if s == 'train' else [r for r in R if r['s'] == s]
    p = np.concatenate([r['oof'] / r['X'][:, 0] for r in rs]); y = np.concatenate([(r['y'] > 0).astype(int) for r in rs])
    print(s, 'AUC(tp>0 by pred rate) %.3f' % auc(p, y), ' '.join('lam%.2f:%+.4f' % (lam, net(rs, lam)[0]) for lam in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]))
imp = full.feature_importance('gain'); names = list(C)[:17] + ['zr', 'rr', 'lrank', 'ct0is0', 'ct1end', 'npred_ntot', 'ntrk']
print(sorted(zip((imp / imp.sum()).round(3), names), reverse=True)[:12])
