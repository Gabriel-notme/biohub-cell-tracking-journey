"""LOEO learnability of candidate-edge costs (evaluable rows only, label = GT edge).
Categories: P15 = existing P15 edge (a cut decision), IN = non-P15 candidate between P15 nodes (swap/link), DROP = touches a dropped detection.
Reports held-out-embryo AUC and the net evaluable edge change of the decisions a cost threshold would make
(cut: removed FP - removed TP; add: pos - neg, ignoring assignment conflicts => optimistic)."""
import os, sys, glob, json
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '4'
sys.path.insert(0, '/workspace/cl/nm')
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import gr_feat
F = gr_feat.FEATS
rows = []
for f in sorted(glob.glob('/workspace/cl/nm/gr_data/*.npz')):
    s, m = os.path.basename(f)[:-4].split('__'); z = np.load(f)
    rows.append((s, m, z['X'], z['pos']))
X = np.concatenate([r[2] for r in rows]); Y = np.concatenate([r[3] for r in rows]).astype(int)
emb = np.concatenate([[r[1][:4]] * len(r[3]) for r in rows]); st = np.concatenate([[r[0]] * len(r[3]) for r in rows])
mov = np.concatenate([[r[1]] * len(r[3]) for r in rows])
p15 = X[:, F.index('p15')] == 1; inn = (~p15) & (X[:, F.index('a_in')] == 1) & (X[:, F.index('b_in')] == 1); drp = (~p15) & ~inn
rad = X[:, F.index('src_rad')] == 1
cats = {'P15': p15, 'IN_cand': inn & ~rad, 'IN_rad': inn & rad, 'DROP_cand': drp & ~rad, 'DROP_rad': drp & rad}
print('rows', len(Y))
for e in ['44b6', '6bba']:
    for c, msk in cats.items():
        mm = msk & (emb == e); print('%s %-10s ev %7d pos %7d (%.3f)' % (e, c, mm.sum(), Y[mm].sum(), Y[mm].mean() if mm.sum() else 0))
P = {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 100, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
     'bagging_freq': 1, 'lambda_l2': 1.0, 'verbose': -1, 'num_threads': 8, 'seed': 0}
pred = np.zeros(len(Y))
os.makedirs('/workspace/cl/nm/gr_models', exist_ok=True)
for te in ['44b6', '6bba']:
    tr = emb != te
    bst = lgb.train(P, lgb.Dataset(X[tr], Y[tr]), num_boost_round=400)
    bst.save_model('/workspace/cl/nm/gr_models/gr_lgb_train_%s.txt' % ('6bba' if te == '44b6' else '44b6'))
    pred[~tr] = bst.predict(X[~tr])
np.save('/workspace/cl/nm/gr_models/loeo_pred.npy', pred)
print('\nLOEO (held-out embryo) per category')
for e in ['44b6', '6bba']:
    for c, msk in cats.items():
        mm = msk & (emb == e)
        if mm.sum() < 10 or Y[mm].min() == Y[mm].max(): continue
        auc = roc_auc_score(Y[mm], pred[mm]); fe = X[mm, F.index('fe')]
        auc_fe = roc_auc_score(Y[mm], fe)
        line = '%s %-10s AUC %.3f (fe-only %.3f) |' % (e, c, auc, auc_fe)
        if c == 'P15':
            for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
                cut = mm & (pred < th); line += ' cut<%.1f: n %d FP %d TP %d net %+d;' % (th, cut.sum(), (Y[cut] == 0).sum(), Y[cut].sum(), (Y[cut] == 0).sum() - Y[cut].sum())
        else:
            for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
                ad = mm & (pred > th); line += ' add>%.1f: n %d pos %d neg %d net %+d;' % (th, ad.sum(), Y[ad].sum(), (Y[ad] == 0).sum(), Y[ad].sum() - (Y[ad] == 0).sum())
        print(line)
# clean40 only
print('\nclean40 (hold36+prev4) held-out embryo')
cl = np.isin(st, ['hold36', 'prev4'])
for e in ['44b6', '6bba']:
    for c, msk in cats.items():
        mm = msk & (emb == e) & cl
        if mm.sum() < 10 or Y[mm].min() == Y[mm].max(): continue
        line = '%s %-10s n %d AUC %.3f |' % (e, c, mm.sum(), roc_auc_score(Y[mm], pred[mm]))
        if c == 'P15':
            for th in [0.2, 0.3, 0.5]:
                cut = mm & (pred < th); line += ' cut<%.1f: FP %d TP %d;' % (th, (Y[cut] == 0).sum(), Y[cut].sum())
        else:
            for th in [0.5, 0.7, 0.9]:
                ad = mm & (pred > th); line += ' add>%.1f: pos %d neg %d;' % (th, Y[ad].sum(), (Y[ad] == 0).sum())
        print(line)
bst = lgb.Booster(model_file='/workspace/cl/nm/gr_models/gr_lgb_train_44b6.txt')
imp = bst.feature_importance('gain'); o = np.argsort(-imp)
print('\ntop features (train 44b6):', [(F[i], int(imp[i])) for i in o[:15]])
