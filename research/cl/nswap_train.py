"""Train node-substitution LightGBM on training-set candidates (evaluable rows, weight = |label|), movie-grouped 5-fold OOF,
report AUC + expected net gain (sum of labels over selected rows, greedy unique n) at thresholds on OOF / hold36 / prev4."""
import sys, json
import numpy as np
import lightgbm as lgb
sys.path.insert(0, '/workspace/cl')
from nswap import FEATS
TRAIN = sys.argv[1].split(',') if len(sys.argv) > 1 else ['b5_t127a', 'b5_t127b', 'b5_audit32']
TEST = sys.argv[2].split(',') if len(sys.argv) > 2 else ['p8_hold36', 'b5_hold36', 'p8_prev4']
OUT = sys.argv[3] if len(sys.argv) > 3 else '/workspace/cl/nswap_lgb.json'


def load(tag):
    z = np.load('/workspace/cl/ns/%s.npz' % tag, allow_pickle=True); return z['X'][:, :-1], z['X'][:, -1], z['movie']


tr = [load(t) for t in TRAIN]
X = np.concatenate([a[0] for a in tr]); y = np.concatenate([a[1] for a in tr]); mv = np.concatenate([a[2] for a in tr])
ev = y != 0
Xe, ye, mve = X[ev], (y[ev] > 0).astype(int), mv[ev]; we = np.abs(y[ev])
print('train evaluable', ev.sum(), 'pos', ye.sum(), 'neg', (1 - ye).sum())
params = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=2.0, verbose=-1, seed=0, num_threads=32)
NR = 500
um = np.unique(mve); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
fe = np.array([fold[m] for m in mve]); oof = np.zeros(len(ye))
for k in range(5):
    b = lgb.train(params, lgb.Dataset(Xe[fe != k], ye[fe != k], weight=we[fe != k]), NR)
    oof[fe == k] = b.predict(Xe[fe == k])
full = lgb.train(params, lgb.Dataset(Xe, ye, weight=we), NR)
json.dump(full.dump_model(), open(OUT, 'w'))


def auc(p, yy):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p)); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


print('OOF AUC %.3f' % auc(oof, ye))
for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    s = oof >= th; print('  OOF th %.1f: selected pos %d (w %d) neg %d (w %d) net %+d' % (th, (s & (ye == 1)).sum(), we[s & (ye == 1)].sum(), (s & (ye == 0)).sum(), we[s & (ye == 0)].sum(), we[s & (ye == 1)].sum() - we[s & (ye == 0)].sum()))
imp = full.feature_importance('gain'); print(sorted(zip((imp / imp.sum()).round(3), FEATS), reverse=True)[:12])
for t in TEST:
    Xt, yt, mt = load(t)
    p = full.predict(Xt)
    e = yt != 0
    print(t, 'AUC %.3f (pos %d neg %d)' % (auc(p[e], (yt[e] > 0).astype(int)), (yt > 0).sum(), (yt < 0).sum()))
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        s = p >= th
        print('  th %.1f: selected %d (pos %d w %d, neg %d w %d, neutral %d) net %+d' % (th, s.sum(), (s & (yt > 0)).sum(), yt[s & (yt > 0)].sum(), (s & (yt < 0)).sum(), -yt[s & (yt < 0)].sum(), (s & (yt == 0)).sum(), yt[s].sum()))
