"""Addition-cost models (non-P15 candidates only: links between P15 nodes and links touching dropped detections).
For each embryo E: a full model trained on E (used for the other embryo = LOEO) and 5 inner movie-fold models on E (OOF, used to
pick the threshold on E only). Variant tag 'noilp' drops the ILP-re-solve features.
usage: gr_train2.py <tag>"""
import os, sys, glob, json
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '4'
sys.path.insert(0, '/workspace/cl/nm')
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import gr_feat
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
F = gr_feat.FEATS
drop_feats = {'noilp': ['ilp_e', 'a_ilp', 'b_ilp']}.get(tag, [])
use = [i for i, f in enumerate(F) if f not in drop_feats]
rows = []
for f in sorted(glob.glob('/workspace/cl/nm/gr_data/*.npz')):
    s, m = os.path.basename(f)[:-4].split('__'); z = np.load(f)
    X = z['X']; k = X[:, F.index('p15')] == 0
    rows.append((s, m, X[k], z['pos'][k]))
X = np.concatenate([r[2] for r in rows]); Y = np.concatenate([r[3] for r in rows]).astype(int)
mov = np.concatenate([[r[1]] * len(r[3]) for r in rows]); emb = np.array([m[:4] for m in mov])
P = {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 15, 'min_data_in_leaf': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
     'bagging_freq': 1, 'lambda_l2': 1.0, 'verbose': -1, 'num_threads': 8, 'seed': 0}
NR = 300
D = '/workspace/cl/nm/gr_models'; os.makedirs(D, exist_ok=True)
fold = {}
oof = np.zeros(len(Y)); loeo = np.zeros(len(Y))
for e in ['44b6', '6bba']:
    me = emb == e
    bst = lgb.train(P, lgb.Dataset(X[me][:, use], Y[me]), num_boost_round=NR)
    bst.save_model('%s/gr2_%s_full_%s.txt' % (D, tag, e))
    loeo[~me] = bst.predict(X[~me][:, use])
    ms = sorted(set(mov[me])); rng = np.random.default_rng(1); rng.shuffle(ms)
    for i, m in enumerate(ms): fold[m] = i % 5
    fm = np.array([fold.get(m, -1) for m in mov])
    for k in range(5):
        tr = me & (fm != k); te = me & (fm == k)
        b = lgb.train(P, lgb.Dataset(X[tr][:, use], Y[tr]), num_boost_round=NR)
        b.save_model('%s/gr2_%s_%s_f%d.txt' % (D, tag, e, k)); oof[te] = b.predict(X[te][:, use])
json.dump({'fold': fold, 'use': [F[i] for i in use]}, open('%s/gr2_%s_meta.json' % (D, tag), 'w'))
drp = (X[:, F.index('a_in')] == 0) | (X[:, F.index('b_in')] == 0)
for e in ['44b6', '6bba']:
    for nm, msk in [('DROP', drp), ('IN', ~drp)]:
        mm = (emb == e) & msk
        line = '%s %-4s n %6d pos %5d | AUC oof %.3f loeo %.3f |' % (e, nm, mm.sum(), Y[mm].sum(), roc_auc_score(Y[mm], oof[mm]), roc_auc_score(Y[mm], loeo[mm]))
        for th in [0.4, 0.5, 0.6, 0.7, 0.8]:
            a = mm & (oof > th); b = mm & (loeo > th)
            line += ' %.1f oof %+d (%d/%d) loeo %+d (%d/%d);' % (th, Y[a].sum() - (Y[a] == 0).sum(), Y[a].sum(), (Y[a] == 0).sum(),
                                                                Y[b].sum() - (Y[b] == 0).sum(), Y[b].sum(), (Y[b] == 0).sum())
        print(line)
for e in ['44b6', '6bba']:
    bst = lgb.Booster(model_file='%s/gr2_%s_full_%s.txt' % (D, tag, e)); imp = bst.feature_importance('gain'); o = np.argsort(-imp)
    print('top features trained on', e, [(F[use[i]], int(imp[i])) for i in o[:12]])
