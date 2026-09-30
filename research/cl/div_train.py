"""Train division-completion acceptance model on B5-base candidates (t127a+t127b+audit32), report on hold36/prev4."""
import json, glob, sys, os
import numpy as np
from multiprocessing import Pool
FEATS = ['is_stolen', 'fork', 'e_old', 'e_pb', 'd_pb', 'd_ab', 'd_pa', 't', 'z', 'hist_p', 'fut_a', 'fut_b', 'cos_ab', 'vel_p', 'dens_p', 'dens_b',
         'ncand_p', 'ncand_b', 'hist_q', 'd_qb', 'd_qp', 'qstart_dp', 'qstart_dt', 'fork_rank_p', 'fork_rank_b', 'fork_gap_p']
NOB1 = [f for f in FEATS if f not in ('fork', 'fork_rank_p', 'fork_rank_b', 'fork_gap_p')]


def load(f):
    rows = json.load(open(f))
    # per-candidate ranks of b1 among competitors
    from collections import defaultdict
    byp = defaultdict(list); byb = defaultdict(list)
    for r in rows: byp[r['p']].append(r['fork']); byb[r['b']].append(r['fork'])
    keep = []
    for r in rows:
        if r['lab'] == 'U' and r['fork'] < 0.3: continue
        sp = sorted(byp[r['p']], reverse=True); sb = sorted(byb[r['b']], reverse=True)
        r['fork_rank_p'] = sp.index(r['fork']); r['fork_rank_b'] = sb.index(r['fork'])
        r['fork_gap_p'] = r['fork'] - (sp[1] if len(sp) > 1 and sp[0] == r['fork'] else sp[0])
        r['is_stolen'] = int(r['typ'] == 'stolen')
        keep.append({k: r.get(k) for k in FEATS + ['lab', 'set', 'movie', 'p', 'a', 'b', 'q', 'typ']})
    return keep


if __name__ == '__main__':
    files = glob.glob('/workspace/cl/cands/*.json')
    with Pool(24) as pool:
        rows = [r for rs in pool.map(load, files) for r in rs]
    json.dump(rows, open('/workspace/cl/div_rows.json', 'w'))
    import lightgbm as lgb
    X = np.array([[(-1 if r.get(k) is None else r[k]) for k in FEATS] for r in rows], dtype=np.float32)
    lab = np.array([r['lab'] for r in rows]); sets = np.array([r['set'] for r in rows]); movies = np.array([r['movie'] for r in rows])
    typ = np.array([r['typ'] for r in rows]); fork = X[:, FEATS.index('fork')]
    y = (lab == 'P').astype(int)
    from collections import Counter
    print(Counter(zip(sets, lab)))
    print('P by type/set', Counter((s, t) for s, t, l in zip(sets, typ, lab) if l == 'P'))
    tr = np.isin(sets, ['t127a', 't127b', 'audit32']) & (lab != 'U')
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=2.0, verbose=-1, seed=0, num_threads=16, scale_pos_weight=1.0)
    def auc(pp, yy):
        if yy.sum() == 0 or yy.sum() == len(yy): return float('nan')
        o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
        return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
    for name, fl in [('full', FEATS), ('nob1', NOB1)]:
        ix = [FEATS.index(f) for f in fl]
        um = np.unique(movies[tr]); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
        oof = np.full(len(rows), np.nan)
        for k in range(5):
            trk = tr & np.array([fold.get(m, -1) != k for m in movies]); vak = tr & np.array([fold.get(m, -1) == k for m in movies])
            oof[vak] = lgb.train(params, lgb.Dataset(X[trk][:, ix], y[trk]), 300).predict(X[vak][:, ix])
        bst = lgb.train(params, lgb.Dataset(X[tr][:, ix], y[tr]), 300); bst.save_model('/workspace/cl/div_lgb_%s.txt' % name)
        p = bst.predict(X[:, ix]); np.save('/workspace/cl/div_p_%s.npy' % name, p)
        print('==', name, 'OOF AUC %.3f' % auc(oof[tr], y[tr]), ' hold36 AUC %.3f  prev4 AUC %.3f' % tuple(auc(p[(sets == s) & (lab != 'U')], y[(sets == s) & (lab != 'U')]) for s in ['hold36', 'prev4']))
        for tt in ['start', 'stolen']:
            m = (typ == tt)
            print('  type', tt, 'OOF AUC %.3f' % auc(oof[tr & m], y[tr & m]), 'hold36 AUC %.3f' % auc(p[m & (sets == 'hold36') & (lab != 'U')], y[m & (sets == 'hold36') & (lab != 'U')]),
                  'b1-only hold36 AUC %.3f' % auc(fork[m & (sets == 'hold36') & (lab != 'U')], y[m & (sets == 'hold36') & (lab != 'U')]))
        for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            line = '  th %.1f' % th
            mo = tr & (oof >= th); line += ' OOF %d/%d' % (y[mo].sum(), mo.sum())
            for s in ['hold36', 'prev4']:
                mm = (sets == s) & (lab != 'U') & (p >= th); line += ' | %s %d/%d' % (s, y[mm].sum(), mm.sum())
            print(line)
    # P3 rule reference
    for s in ['t127a', 't127b', 'audit32', 'hold36', 'prev4']:
        m = (sets == s) & (lab != 'U') & (((typ == 'start') & (fork >= 0.9)) | ((typ == 'stolen') & (fork >= 0.97)))
        print('P3 rule', s, '%d/%d' % (y[m].sum(), m.sum()), ' total P', y[(sets == s)].sum())
