"""Cross-set models: train edge_link and relink on base training sets + one extra validation set.
usage: train_xset.py <tag> <extra_set or none>"""
import json, glob, sys
sys.path.insert(0, '/workspace/cl')
import numpy as np, lightgbm as lgb
import edge_link, relink
tag, extra = sys.argv[1], sys.argv[2]
TR = ['t127a', 't127b', 'audit32'] + ([] if extra == 'none' else extra.split(','))
params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=8)
for name, feats, cdir in [('edge', edge_link.FEATS, '/workspace/cl/ecands'), ('relink', relink.FEATS, '/workspace/cl/rcands')]:
    rows = []
    for s in TR:
        for f in glob.glob('%s/%s__*.json' % (cdir, s)): rows += json.load(open(f))
    rows = [r for r in rows if r['lab'] != 'U']
    X = np.array([[(-1 if r.get(k) is None else r.get(k, -1)) for k in feats] for r in rows], dtype=np.float32)
    y = np.array([r['lab'] == 'P' for r in rows], dtype=int)
    b = lgb.train(params, lgb.Dataset(X, y), 400)
    b.save_model('/workspace/cl/%s_lgb_%s.txt' % (name, tag)); json.dump(b.dump_model(), open('/workspace/cl/%s_lgb_%s.json' % (name, tag), 'w'))
    print(tag, name, 'trained on', TR, 'n', len(rows), 'pos', int(y.sum()))
