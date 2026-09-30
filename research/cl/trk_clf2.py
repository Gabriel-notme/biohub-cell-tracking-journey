"""Leak-free version of the tracklet-removal test. The decision may only use test-time information:
break-even per node uses a constant TP estimate (train median) and ntot estimated as npred * c (train median ratio).
Features: only prediction-side quantities. True gain is then computed with the real TP/ntot."""
import glob
import numpy as np
import lightgbm as lgb
C = dict(len=0, t0=1, t1=2, st=3, en=4, z=5, y=6, x=7, sz=8, sy=9, sx=10, v=11, vmax=12, csz=13, ct0=14, ct1=15, ep=16, nm=17, tp=18)
R = []
for f in sorted(glob.glob('/workspace/cl/trk/*.npz')):
    s, m = f.split('/')[-1][:-4].split('__')
    z = np.load(f); X = z['X']; ntot = float(z['n_total']); npred = int(z['n_pred']); nge = int(z['n_gt_edges'])
    TP = X[:, C['tp']].sum(); mi = 1 - 0.1 * (npred - ntot) / ntot
    zr = (X[:, C['z']] - X[:, C['z']].min()) / max(1e-6, np.ptp(X[:, C['z']]))
    rr = np.sqrt((X[:, C['y']] - 128) ** 2 + (X[:, C['x']] - 128) ** 2)
    lrank = np.argsort(np.argsort(X[:, 0])) / len(X)
    F = np.column_stack([X[:, :17], zr, rr, lrank, (X[:, C['ct0']] == 0), (X[:, C['ct1']] >= 99), np.full(len(X), npred), np.full(len(X), len(X))])
    R.append(dict(s=s, m=m, X=X, F=F, y=X[:, C['tp']], TP=TP, mi=mi, ntot=ntot, npred=npred, W=nge))
TR = [r for r in R if r['s'] in ('t127a', 't127b', 'audit32')]
TPmed = np.median([r['TP'] for r in TR]); ratio = np.median([r['ntot'] / r['npred'] for r in TR])
print('train median TP %.0f, ntot/npred %.3f' % (TPmed, ratio))
for r in R:
    ntot_hat = r['npred'] * ratio
    r['be_hat'] = 0.1 * TPmed / ntot_hat / (1 - 0.1 * (r['npred'] - ntot_hat) / ntot_hat)
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
    g = l = W = 0.; nn = kk = 0
    for r in rs:
        sel = r['oof'] < lam * r['X'][:, 0] * r['be_hat']
        n = r['X'][sel, 0].sum(); k = r['y'][sel].sum()
        g += r['TP'] * 0.1 * n / r['ntot']; l += k * r['mi']; W += r['W']; nn += n; kk += k
    return (g - l) / W, nn, kk


for s in ['train', 'hold36', 'prev4']:
    rs = TR if s == 'train' else [r for r in R if r['s'] == s]
    print(s, ' '.join('lam%.2f:%+.5f(n%d,k%d)' % ((lam,) + net(rs, lam)) for lam in [0.2, 0.3, 0.5, 0.7, 1.0]))
full.save_model('/workspace/cl/trk_prune_lgb.txt')
