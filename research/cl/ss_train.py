import json, glob, sys
import numpy as np, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, '/workspace/cl')
import swap
FS = {'steal': ['d_sd', 'd_pd', 'm_sd', 'm_pd', 'd_sp', 'hist_s', 'hist_p', 'fut_d', 'fe_sd', 'fe_pd', 'ep_pd', 'sp_s', 'sp_p', 'z', 'b1_sd', 'b1_pd', 'b1_diff'],
      'swap': swap.FEATS + ['b1_c1', 'b1_c2', 'b1_a1', 'b1_a2', 'b1_diff']}
D = {'steal': [], 'swap': []}
for f in glob.glob('/workspace/cl/ssc/*.json'):
    s, m = Path(f).stem.split('__'); d = json.load(open(f))
    for k in D:
        for r in d[k]: r['set'] = s; r['movie'] = m; D[k].append(r)
params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=0, num_threads=16)


def auc(pp, yy):
    o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
    return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


for k, rows in D.items():
    fe = FS[k]
    X = np.array([[(-1 if r.get(c) is None else r.get(c, -1)) for c in fe] for r in rows], np.float32)
    lab = np.array([r['lab'] for r in rows]); sets = np.array([r['set'] for r in rows]); mv = np.array([r['movie'] for r in rows]); gain = np.array([r['gain'] for r in rows])
    y = (lab == 'P').astype(int)
    tr = np.isin(sets, ['t127a', 't127b', 'audit32']) & (lab != 'U')
    um = np.unique(mv[tr]); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
    oof = np.full(len(rows), np.nan); trall = np.isin(sets, ['t127a', 't127b', 'audit32'])
    for f_ in range(5):
        trk = tr & np.array([fold.get(m, -1) != f_ for m in mv]); vak = trall & np.array([fold.get(m, -1) == f_ for m in mv])
        oof[vak] = lgb.train(params, lgb.Dataset(X[trk], y[trk]), 400).predict(X[vak])
    b = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 400)
    json.dump(b.dump_model(), open('/workspace/cl/%s_b1_lgb.json' % k, 'w'))
    p = b.predict(X)
    print('==', k, 'rows', len(rows), 'train P/N %d/%d' % (y[tr].sum(), (tr & (y == 0)).sum()), 'OOF AUC %.3f' % auc(oof[tr], y[tr]),
          ' '.join('%s AUC %.3f (P %d N %d)' % (s, auc(p[(sets == s) & (lab != 'U')], y[(sets == s) & (lab != 'U')]), y[(sets == s)].sum(), ((sets == s) & (lab == 'N')).sum()) for s in ['hold36', 'prev4']))
    print('   b1_diff-only AUC hold36 %.3f' % auc(X[(sets == 'hold36') & (lab != 'U'), fe.index('b1_diff')], y[(sets == 'hold36') & (lab != 'U')]))
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mo = trall & (oof >= th); line = '  th %.1f OOF n %d P %d N %d gain %+.1f' % (th, mo.sum(), y[mo].sum(), (mo & (lab == 'N')).sum(), gain[mo].sum())
        for s in ['hold36', 'prev4']:
            mm = (sets == s) & (p >= th); line += ' | %s n %d P %d N %d gain %+.1f' % (s, mm.sum(), y[mm].sum(), (mm & (lab == 'N')).sum(), gain[mm].sum())
        print(line)
