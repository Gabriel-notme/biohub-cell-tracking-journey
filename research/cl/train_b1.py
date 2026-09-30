"""Retrain edge_link and relink models with b1 edge-head probabilities as extra features; compare with the originals."""
import json, glob, sys
sys.path.insert(0, '/workspace/cl')
import numpy as np, lightgbm as lgb
import edge_link, relink
b1 = {}
for f in glob.glob('/workspace/cl/b1e/*.json'):
    s, m = f.split('/')[-1][:-5].split('__'); b1[(s, m)] = json.load(open(f))


def g(s, m, a, b):
    if a is None or b is None: return -2.
    return b1[(s, m)].get('%d_%d' % (a, b), -1.)


params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=8)


def auc(pp, yy):
    o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
    return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


def run(name, rows, feats, out):
    X = np.array([[(-1 if r.get(k) is None else r.get(k, -1)) for k in feats] for r in rows], dtype=np.float32)
    lab = np.array([r['lab'] for r in rows]); sets = np.array([r['set'] for r in rows]); movies = np.array([r['movie'] for r in rows])
    y = (lab == 'P').astype(int)
    tr = np.isin(sets, ['t127a', 't127b', 'audit32']) & (lab != 'U')
    um = np.unique(movies[tr]); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
    oof = np.full(len(rows), np.nan)
    for k in range(5):
        trk = tr & np.array([fold.get(m, -1) != k for m in movies]); vak = tr & np.array([fold.get(m, -1) == k for m in movies])
        oof[vak] = lgb.train(params, lgb.Dataset(X[trk], y[trk]), 400).predict(X[vak])
    bst = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 400)
    if out:
        bst.save_model('/workspace/cl/%s.txt' % out); json.dump(bst.dump_model(), open('/workspace/cl/%s.json' % out, 'w'))
    p = bst.predict(X)
    print('==', name, 'OOF AUC %.3f' % auc(oof[tr], y[tr]), ' '.join('%s AUC %.3f' % (s, auc(p[(sets == s) & (lab != 'U')], y[(sets == s) & (lab != 'U')])) for s in ['hold36', 'prev4']))
    for th in [0.3, 0.4, 0.5, 0.65, 0.8]:
        mo = tr & (oof >= th); line = '  th %.2f OOF %d/%d' % (th, y[mo].sum(), mo.sum())
        for s in ['hold36', 'prev4']:
            mm = (sets == s) & (lab != 'U') & (p >= th); line += ' | %s %d/%d of %d' % (s, y[mm].sum(), mm.sum(), y[(sets == s) & (lab != 'U')].sum())
        print(line)


er = []
for f in glob.glob('/workspace/cl/ecands/*.json'): er += json.load(open(f))
for r in er: r['b1e'] = g(r['set'], r['movie'], r['s'], r['d']) if r['gap'] == 1 else -2.
run('edge_link orig', er, edge_link.FEATS, None)
run('edge_link +b1', er, edge_link.FEATS + ['b1e'], 'edge_lgb_b1')
rr = []
for f in glob.glob('/workspace/cl/rcands/*.json'): rr += json.load(open(f))
for r in rr:
    r['b1_sd'] = g(r['set'], r['movie'], r['s'], r['d']); r['b1_scur'] = g(r['set'], r['movie'], r['s'], r['cur_d']); r['b1_curd'] = g(r['set'], r['movie'], r['cur_s'], r['d'])
    r['b1_diff_s'] = r['b1_sd'] - r['b1_scur'] if r['cur_d'] is not None else 1.0
    r['b1_diff_d'] = r['b1_sd'] - r['b1_curd'] if r['cur_s'] is not None else 1.0
run('relink orig', rr, relink.FEATS, None)
run('relink +b1', rr, relink.FEATS + ['b1_sd', 'b1_scur', 'b1_curd', 'b1_diff_s', 'b1_diff_d'], 'relink_lgb_b1')
