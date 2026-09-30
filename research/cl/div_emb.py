"""Division completion from b1 node embeddings: GBM on [emb(p), emb(a), emb(b), emb(q), b1 fork prob, geometry] trained on the
candidate distribution itself (corrected labels: P = new GT division, D/N = negative). Train t127a+t127b+audit32, test hold36/prev4."""
import json, glob
import numpy as np
from pathlib import Path
from multiprocessing import Pool
GEO = ['fork', 'e_old', 'e_pb', 'd_pb', 'd_ab', 'd_pa', 'hist_p', 'fut_a', 'fut_b', 'cos_ab', 'vel_p', 'dens_p', 'dens_b', 'hist_q', 'd_qb', 'd_qp', 'qstart_dp', 'qstart_dt']


def load(f):
    s, m = Path(f).stem.split('__')
    ef = '/workspace/cl/emb/%s__%s.npz' % (s, m)
    if not Path(ef).exists(): return []
    z = np.load(ef); ids = z['ids']; E = z['emb'].astype(np.float32); kind = z['kind']
    look = {int(i): j for j, (i, k) in enumerate(zip(ids.tolist(), kind.tolist())) if k == 0}
    rows = json.load(open(f)); labs = json.load(open(f.replace('/cands/', '/clab/')))
    rng = np.random.default_rng(abs(hash(m)) % (2 ** 31))
    out = []
    for r, l in zip(rows, labs):
        if l == 'U': continue
        if l == 'N' and r['fork'] < 0.3 and rng.random() > 0.1: continue  # subsample easy negatives
        ix = [look.get(r['p']), look.get(r['a']), look.get(r['b'])]
        if any(i is None for i in ix): continue
        q = look.get(r['q']) if r['q'] is not None else None
        v = np.concatenate([E[ix[0]], E[ix[1]], E[ix[2]], E[q] if q is not None else np.zeros(E.shape[1], np.float32),
                            np.array([(-1 if r.get(k) is None else r[k]) for k in GEO] + [int(r['typ'] == 'stolen')], np.float32)])
        w = 1.0 if (l != 'N' or r['fork'] >= 0.3) else 10.0
        out.append((s, m, l, r['typ'], float(r['fork']), w, v))
    return out


if __name__ == '__main__':
    with Pool(24) as pool: R = [x for xs in pool.map(load, glob.glob('/workspace/cl/cands/*.json')) for x in xs]
    import lightgbm as lgb
    sets = np.array([x[0] for x in R]); mv = np.array([x[1] for x in R]); lab = np.array([x[2] for x in R]); typ = np.array([x[3] for x in R])
    b1 = np.array([x[4] for x in R]); w = np.array([x[5] for x in R]); X = np.stack([x[6] for x in R]); y = (lab == 'P').astype(int)
    print('rows', len(R), 'dims', X.shape[1], {s: (int(y[sets == s].sum()), int((sets == s).sum())) for s in np.unique(sets)})
    tr = np.isin(sets, ['t127a', 't127b', 'audit32'])
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=0, num_threads=24)
    um = np.unique(mv[tr]); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
    oof = np.full(len(R), np.nan)
    for k in range(5):
        trk = tr & np.array([fold.get(m, -1) != k for m in mv]); vak = tr & np.array([fold.get(m, -1) == k for m in mv])
        oof[vak] = lgb.train(params, lgb.Dataset(X[trk], y[trk], weight=w[trk]), 300).predict(X[vak])
    b = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr]), 300); p = b.predict(X)
    def auc(pp, yy, ww=None):
        o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
        return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)
    for t_ in ['start', 'stolen']:
        m_tr = tr & (typ == t_); m_h = (sets == 'hold36') & (typ == t_)
        print(t_, 'OOF AUC model %.3f b1 %.3f | hold36 AUC model %.3f b1 %.3f (P %d)' % (auc(oof[m_tr], y[m_tr]), auc(b1[m_tr], y[m_tr]), auc(p[m_h], y[m_h]), auc(b1[m_h], y[m_h]), y[m_h].sum()))
    for th in [0.1, 0.2, 0.3, 0.5, 0.7]:
        line = 'th %.1f' % th
        for nm, sc, msk in [('OOF', oof, tr), ('hold36', p, sets == 'hold36'), ('prev4', p, sets == 'prev4')]:
            k = msk & (sc >= th); line += ' | %s P %d D %d N %d' % (nm, (k & (lab == 'P')).sum(), (k & (lab == 'D')).sum(), (k & (lab == 'N')).sum())
        print(line)
    for nm, msk in [('train', tr), ('hold36', sets == 'hold36')]:
        k = msk & (((typ == 'start') & (b1 >= 0.9)) | ((typ == 'stolen') & (b1 >= 0.97)))
        print('P3 rule', nm, 'P %d D %d N %d' % ((k & (lab == 'P')).sum(), (k & (lab == 'D')).sum(), (k & (lab == 'N')).sum()))
