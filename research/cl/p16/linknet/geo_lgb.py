"""Geometry-only (rl_cands FEATS) and geometry+b1 LightGBM LOEO predictions on the rl_cands evaluable rows (baseline for linknet).
-> geo_pred.pkl {movie: {'geo': p, 'geob1': p}}"""
import os, sys, glob, pickle
import numpy as np, lightgbm as lgb
sys.path.insert(0, '/workspace/cl/nm')
from rl_cands import FEATS
M = []
for f in sorted(glob.glob('/workspace/cl/nm/rl_cands/*.pkl')):
    s, m = os.path.basename(f)[:-4].split('__'); z = pickle.load(open(f, 'rb')); e = pickle.load(open('/workspace/cl/nm/rl_b1f/' + os.path.basename(f), 'rb'))
    L = z['L']; y = ((L['tp_sd'] == 1) & (L['tp_sc'] == 0) & (L['tp_qd'] == 0)).astype(int)
    M.append(dict(movie=m, emb=m[:4], X=z['X'], B=e['X'], y=y))
P = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, seed=0, num_threads=32)
out = {m['movie']: {} for m in M}
for tr, te in [('44b6', '6bba'), ('6bba', '44b6')]:
    mtr = [m for m in M if m['emb'] == tr]; mte = [m for m in M if m['emb'] == te]
    for key, fx in [('geo', lambda m: m['X']), ('geob1', lambda m: np.column_stack([m['X'], m['B']]))]:
        b = lgb.train(P, lgb.Dataset(np.concatenate([fx(m) for m in mtr]), np.concatenate([m['y'] for m in mtr])), 500)
        for m in mte: out[m['movie']][key] = b.predict(fx(m))
        print(tr, '->', te, key, 'done', flush=True)
pickle.dump(out, open('/workspace/cl/p16/linknet/geo_pred.pkl', 'wb'))
