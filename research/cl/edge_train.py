"""Train free-end linking classifier on t127a+t127b+audit32 labeled candidates; report precision/recall on hold36/prev4."""
import json, glob, sys
import numpy as np
import lightgbm as lgb
from pathlib import Path
FEATS = ['gap', 'dist', 'dz', 'dxy', 'hist_s', 'fut_d', 'sp_s', 'sp_d', 'dpred', 'dpred_d', 'cos_s', 'cos_d', 'z', 'tt', 'dens_s', 'dens_d',
         'nn_d_prev', 'nn_s_next', 'fe', 'drop_mid', 'n_s', 'n_d', 'rk_s', 'rk_d', 'gap_s', 'gap_d']
TRAIN = {'t127a', 't127b', 'audit32'}
rows = []
for f in glob.glob('/workspace/cl/ecands/*.json'):
    rows += json.load(open(f))
X = np.array([[r.get(k, -1) for k in FEATS] for r in rows], dtype=np.float32)
lab = np.array([r['lab'] for r in rows]); sets = np.array([r['set'] for r in rows]); movies = np.array([r['movie'] for r in rows])
tr = np.isin(sets, list(TRAIN)) & (lab != 'U')
y = (lab == 'P').astype(int)
# grouped CV on training movies for threshold selection
um = np.unique(movies[tr]); rng = np.random.default_rng(0); rng.shuffle(um)
fold = {m: i % 5 for i, m in enumerate(um)}
oof = np.full(len(rows), np.nan)
params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=8)
NR = 400
for k in range(5):
    trk = tr & np.array([fold.get(m, -1) != k for m in movies]); vak = tr & np.array([fold.get(m, -1) == k for m in movies])
    b = lgb.train(params, lgb.Dataset(X[trk], y[trk]), NR)
    oof[vak] = b.predict(X[vak])
bst = lgb.train(params, lgb.Dataset(X[tr], y[tr]), NR)
bst.save_model('/workspace/cl/edge_lgb.txt')
p_all = bst.predict(X)
np.save('/workspace/cl/edge_p_all.npy', p_all)
def auc(p, yy):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p)); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
print('OOF AUC train %.3f' % auc(oof[tr], y[tr]))
for s in ['hold36', 'prev4']:
    m = (sets == s) & (lab != 'U'); print(s, 'AUC %.3f' % auc(p_all[m], y[m]), 'P', y[m].sum(), 'N', (1 - y[m]).sum())
print('thr   OOF_prec OOF_rec  |  hold36 prec rec  (P,N)  | prev4 prec rec')
for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    mo = tr & (oof >= th); line = '%.2f  %.3f  %.3f (%d/%d) ' % (th, y[mo].mean() if mo.sum() else 0, (y[mo].sum() / y[tr].sum()), y[mo].sum(), mo.sum())
    for s in ['hold36', 'prev4']:
        m = (sets == s) & (lab != 'U'); mm = m & (p_all >= th)
        line += '| %s %.3f %.3f (%d,%d) ' % (s, y[mm].mean() if mm.sum() else 0, y[mm].sum() / max(1, y[m].sum()), y[mm].sum(), mm.sum() - y[mm].sum())
    print(line)
imp = sorted(zip(bst.feature_importance('gain'), FEATS), reverse=True)
print('importance', [(f, int(g)) for g, f in imp[:12]])
