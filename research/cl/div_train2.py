"""Retrain division acceptance model with corrected labels (P only for not-yet-recovered GT divisions; D = duplicate -> negative)."""
import json, glob, sys
import numpy as np
from multiprocessing import Pool
from collections import defaultdict, Counter
FEATS = ['is_stolen', 'fork', 'e_old', 'e_pb', 'd_pb', 'd_ab', 'd_pa', 't', 'z', 'hist_p', 'fut_a', 'fut_b', 'cos_ab', 'vel_p', 'dens_p', 'dens_b',
         'ncand_p', 'ncand_b', 'hist_q', 'd_qb', 'd_qp', 'qstart_dp', 'qstart_dt', 'fork_rank_p', 'fork_rank_b', 'fork_gap_p']
NOB1 = [f for f in FEATS if f not in ('fork', 'fork_rank_p', 'fork_rank_b', 'fork_gap_p')]


def load(f):
    rows = json.load(open(f)); labs = json.load(open(f.replace('/cands/', '/clab/')))
    byp = defaultdict(list); byb = defaultdict(list)
    for r in rows: byp[r['p']].append(r['fork']); byb[r['b']].append(r['fork'])
    keep = []
    for r, l in zip(rows, labs):
        if l == 'U' and r['fork'] < 0.3: continue
        sp = sorted(byp[r['p']], reverse=True); sb = sorted(byb[r['b']], reverse=True)
        r['fork_rank_p'] = sp.index(r['fork']); r['fork_rank_b'] = sb.index(r['fork'])
        r['fork_gap_p'] = r['fork'] - (sp[1] if len(sp) > 1 and sp[0] == r['fork'] else sp[0])
        r['is_stolen'] = int(r['typ'] == 'stolen'); r['lab'] = l
        keep.append({k: r.get(k) for k in FEATS + ['lab', 'set', 'movie', 'typ']})
    return keep


if __name__ == '__main__':
    with Pool(24) as pool:
        rows = [r for rs in pool.map(load, glob.glob('/workspace/cl/cands/*.json')) for r in rs]
    import lightgbm as lgb
    X = np.array([[(-1 if r.get(k) is None else r[k]) for k in FEATS] for r in rows], dtype=np.float32)
    lab = np.array([r['lab'] for r in rows]); sets = np.array([r['set'] for r in rows]); movies = np.array([r['movie'] for r in rows])
    typ = np.array([r['typ'] for r in rows]); fork = X[:, FEATS.index('fork')]
    y = (lab == 'P').astype(int); labd = lab != 'U'
    tr = np.isin(sets, ['t127a', 't127b', 'audit32']) & labd
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=2.0, verbose=-1, seed=0, num_threads=16)

    def auc(pp, yy):
        if yy.sum() == 0 or yy.sum() == len(yy): return float('nan')
        o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
        return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
    for name, fl in [('full2', FEATS), ('nob12', NOB1)]:
        ix = [FEATS.index(f) for f in fl]
        um = np.unique(movies[tr]); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
        oof = np.full(len(rows), np.nan)
        for k in range(5):
            trk = tr & np.array([fold.get(m, -1) != k for m in movies]); vak = tr & np.array([fold.get(m, -1) == k for m in movies])
            oof[vak] = lgb.train(params, lgb.Dataset(X[trk][:, ix], y[trk]), 300).predict(X[vak][:, ix])
        bst = lgb.train(params, lgb.Dataset(X[tr][:, ix], y[tr]), 300)
        bst.save_model('/workspace/cl/div_lgb_%s.txt' % name)
        p = bst.predict(X[:, ix])
        print('==', name, 'OOF AUC %.3f' % auc(oof[tr], y[tr]), ' hold36 AUC %.3f  prev4 AUC %.3f' % tuple(auc(p[(sets == s) & labd], y[(sets == s) & labd]) for s in ['hold36', 'prev4']))
        for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            line = '  th %.1f' % th
            mo = tr & (oof >= th); line += ' OOF P %d D %d N %d' % (y[mo].sum(), (lab[mo] == 'D').sum(), (lab[mo] == 'N').sum())
            for s in ['hold36', 'prev4']:
                mm = (sets == s) & labd & (p >= th); line += ' | %s P %d D %d N %d' % (s, y[mm].sum(), (lab[mm] == 'D').sum(), (lab[mm] == 'N').sum())
            print(line)
    for s in ['t127a', 't127b', 'audit32', 'hold36', 'prev4']:
        m = (sets == s) & labd & (((typ == 'start') & (fork >= 0.9)) | ((typ == 'stolen') & (fork >= 0.97)))
        print('P3 rule', s, 'P %d D %d N %d' % (y[m].sum(), (lab[m] == 'D').sum(), (lab[m] == 'N').sum()), ' total P', y[(sets == s)].sum())
