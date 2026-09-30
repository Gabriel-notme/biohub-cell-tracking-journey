"""clean40: does b1's edge head add cost signal? AUC of b1 prob vs the geometric LOEO model per category, and cross-embryo
LightGBM on [geometric features + b1] trained on one embryo's clean movies, tested on the other (tiny but b1-clean)."""
import os, sys, glob
sys.path.insert(0, '/workspace/cl/nm')
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import gr_feat
F = gr_feat.FEATS
files = sorted(glob.glob('/workspace/cl/nm/gr_data/*.npz'))
loeo = np.load('/workspace/cl/nm/gr_models/loeo_pred.npy')
off = 0; R = []
for f in files:
    s, m = os.path.basename(f)[:-4].split('__'); z = np.load(f); n = len(z['pos'])
    if s in ('hold36', 'prev4'):
        b1 = np.load('/workspace/cl/nm/gr_b1/%s__%s.npy' % (s, m))
        R.append((m, z['X'], z['pos'].astype(int), loeo[off:off + n], b1))
    off += n
X = np.concatenate([r[1] for r in R]); Y = np.concatenate([r[2] for r in R]); G = np.concatenate([r[3] for r in R]); B = np.concatenate([r[4] for r in R])
emb = np.concatenate([[r[0][:4]] * len(r[2]) for r in R])
p15 = X[:, F.index('p15')] == 1; inn = (~p15) & (X[:, F.index('a_in')] == 1) & (X[:, F.index('b_in')] == 1)
rad = X[:, F.index('src_rad')] == 1
cats = {'P15': p15, 'IN_cand': inn & ~rad, 'IN_rad': inn & rad, 'DROP': ~p15 & ~inn}
for e in ['44b6', '6bba']:
    for c, msk in cats.items():
        mm = msk & (emb == e)
        if mm.sum() < 20 or Y[mm].min() == Y[mm].max(): continue
        line = '%s %-8s n %6d pos %5d | AUC geo-LOEO %.3f b1 %.3f |' % (e, c, mm.sum(), Y[mm].sum(), roc_auc_score(Y[mm], G[mm]), roc_auc_score(Y[mm], B[mm]))
        if c == 'P15':
            for th in [0.05, 0.1, 0.2, 0.3]:
                k = mm & (B < th); line += ' b1<%.2f FP %d TP %d;' % (th, (Y[k] == 0).sum(), Y[k].sum())
        else:
            for th in [0.5, 0.7, 0.9, 0.95]:
                k = mm & (B > th); line += ' b1>%.2f pos %d neg %d;' % (th, Y[k].sum(), (Y[k] == 0).sum())
        print(line)
# cross-embryo combiner on clean40 only
P = {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 15, 'min_data_in_leaf': 50, 'verbose': -1, 'num_threads': 8, 'seed': 0}
Xb = np.column_stack([X, B])
for te in ['44b6', '6bba']:
    tr = emb != te
    for lab, XX in [('geo', X), ('geo+b1', Xb)]:
        bst = lgb.train(P, lgb.Dataset(XX[tr], Y[tr]), num_boost_round=300); p = bst.predict(XX[~tr])
        line = 'train %s -> test %s %-7s' % ('6bba' if te == '44b6' else '44b6', te, lab)
        for c, msk in cats.items():
            mm = msk[~tr]; yy = Y[~tr][mm]
            if mm.sum() < 20 or yy.min() == yy.max(): continue
            pp = p[mm]
            if c == 'P15': k = pp < 0.5; line += ' | %s AUC %.3f cut<.5 FP %d TP %d' % (c, roc_auc_score(yy, pp), (yy[k] == 0).sum(), yy[k].sum())
            else: k = pp > 0.5; line += ' | %s AUC %.3f add>.5 pos %d neg %d' % (c, roc_auc_score(yy, pp), yy[k].sum(), (yy[k] == 0).sum())
        print(line)
