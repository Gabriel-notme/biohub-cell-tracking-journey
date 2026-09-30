"""LOEO AUC of relink models with / without b1 edge-head features (labelled rows only), cross-embryo direction both ways."""
import glob, pickle
import numpy as np
import lightgbm as lgb
D = []
for f in glob.glob('/workspace/cl/lo/rb1/*.pkl'):
    s, m = f.split('/')[-1][:-4].split('__'); z = pickle.load(open(f, 'rb'))
    if len(z['y']): D.append((s, m, z['X'], z['B'], z['y']))
print('movies', len(D), 'labelled rows', sum((d[4] >= 0).sum() for d in D), 'pos', sum((d[4] == 1).sum() for d in D))
P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=16)


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p)); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


def stack(ds, use_b1):
    X = np.concatenate([np.column_stack([d[2], d[3]]) if use_b1 else d[2] for d in ds]); y = np.concatenate([d[4] for d in ds]); k = y >= 0
    return X[k], y[k]


for src, dst in [('44b6', '6bba'), ('6bba', '44b6')]:
    for use_b1 in [False, True]:
        Xa, ya = stack([d for d in D if d[1].startswith(src)], use_b1); Xb, yb = stack([d for d in D if d[1].startswith(dst)], use_b1)
        b = lgb.train(P, lgb.Dataset(Xa, ya), 400); p = b.predict(Xb)
        print('train %s -> test %s  b1=%s  AUC %.3f  (test pos %d / %d)  b1_sd-alone AUC %.3f' % (src, dst, use_b1, auc(p, yb), yb.sum(), len(yb), auc(Xb[:, -5], yb) if use_b1 else float('nan')))
