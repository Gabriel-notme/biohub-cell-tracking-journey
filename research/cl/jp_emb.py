"""Train-only: junk-prune with a per-embryo tp_ref (median train TP per embryo) vs the global tp_ref. Same OOF folds as jp_robust."""
import pickle
import numpy as np
import lightgbm as lgb
TRAIN = ['p8_t127a', 'p8_t127b', 'p8_audit32']
TR = []
for t in TRAIN:
    for r in pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb')): r['set'] = t; TR.append(r)
G_REF = float(np.median([r['TP'] for r in TR])); RATIO = float(np.median([r['ntot'] / r['npred'] for r in TR]))
E_REF = {e: float(np.median([r['TP'] for r in TR if r['movie'].startswith(e)])) for e in ['44b6', '6bba']}
print('global tp_ref', G_REF, 'per-embryo', E_REF)
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=32)
rng = np.random.default_rng(0); idx = np.arange(len(TR)); rng.shuffle(idx); fold = {int(i): k % 5 for k, i in enumerate(idx)}
for k in range(5):
    a = [r for i, r in enumerate(TR) if fold[i] != k]; b = [r for i, r in enumerate(TR) if fold[i] == k]
    m = lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in a]), np.concatenate([r['tp'] for r in a])), 300)
    for r in b: r['pred'] = m.predict(r['X'])


def parts(r, lam, ref):
    m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
    ntot_hat = r['npred'] * RATIO; m_hat = 1 - 0.1 * (r['npred'] - ntot_hat) / ntot_hat; be = 0.1 * ref / ntot_hat / m_hat
    sel = r['pred'] < lam * r['len'] * be
    n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum()
    return r['TP'] * m, w, (r['TP'] - k) * (m + 0.1 * n / r['ntot']), w - f


def gain(rs, lam, mode):
    P = np.array([parts(r, lam, G_REF if mode == 'global' else E_REF[r['movie'][:4]]) for r in rs])
    return P[:, 2].sum() / P[:, 3].sum() - P[:, 0].sum() / P[:, 1].sum()


groups = {}
for r in TR: groups.setdefault((r['set'], r['movie'][:4]), []).append(r)
for mode in ['global', 'embryo']:
    for lam in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        g = {k: gain(v, lam, mode) for k, v in groups.items()}
        print('%-6s lam %.2f all %+.5f min %+.5f | %s' % (mode, lam, gain(TR, lam, mode), min(g.values()), ' '.join('%s/%s %+.5f' % (k[0][3:], k[1], v) for k, v in sorted(g.items()))))
