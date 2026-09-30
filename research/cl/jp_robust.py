"""Train-only robustness of the junk-prune lam: OOF gain per train subgroup (set x embryo) and bootstrap over train movies.
Pre-registered choice: lam maximising the minimum subgroup OOF gain (ties -> smaller lam)."""
import sys, pickle
import numpy as np
import lightgbm as lgb
TRAIN = ['p8_t127a', 'p8_t127b', 'p8_audit32']
TR = []
for t in TRAIN:
    for r in pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb')): r['set'] = t; TR.append(r)
TP_REF = float(np.median([r['TP'] for r in TR])); RATIO = float(np.median([r['ntot'] / r['npred'] for r in TR]))
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=32)
rng = np.random.default_rng(0); idx = np.arange(len(TR)); rng.shuffle(idx); fold = {int(i): k % 5 for k, i in enumerate(idx)}
for k in range(5):
    a = [r for i, r in enumerate(TR) if fold[i] != k]; b = [r for i, r in enumerate(TR) if fold[i] == k]
    m = lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in a]), np.concatenate([r['tp'] for r in a])), 300)
    for r in b: r['pred'] = m.predict(r['X']) if len(r['X']) else np.zeros(0)


def parts(r, lam):
    m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
    ntot_hat = r['npred'] * RATIO; m_hat = 1 - 0.1 * (r['npred'] - ntot_hat) / ntot_hat; be = 0.1 * TP_REF / ntot_hat / m_hat
    sel = r['pred'] < lam * r['len'] * be if len(r['len']) else np.zeros(0, bool)
    n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum()
    return r['TP'] * m, w, (r['TP'] - k) * (m + 0.1 * n / r['ntot']), w - f


def gain(rs, lam):
    P = np.array([parts(r, lam) for r in rs]); return P[:, 2].sum() / P[:, 3].sum() - P[:, 0].sum() / P[:, 1].sum()


LAMS = [0.5, 0.75, 1.0, 1.25]
groups = {}
for r in TR: groups.setdefault((r['set'], r['movie'][:4]), []).append(r)
print('subgroup OOF gains:')
mins = {}
for lam in LAMS:
    g = {k: gain(v, lam) for k, v in groups.items()}
    mins[lam] = min(g.values())
    print('  lam %.2f all %+.5f  min %+.5f | ' % (lam, gain(TR, lam), mins[lam]) + ' '.join('%s/%s %+.5f' % (k[0][3:], k[1], v) for k, v in sorted(g.items())))
B = 500; brng = np.random.default_rng(1)
bs = {lam: [] for lam in LAMS}
for _ in range(B):
    ii = brng.integers(0, len(TR), len(TR)); rs = [TR[i] for i in ii]
    for lam in LAMS: bs[lam].append(gain(rs, lam))
for lam in LAMS:
    v = np.array(bs[lam]); print('  lam %.2f bootstrap over train movies: mean %+.5f 5%% %+.5f P>0 %.3f' % (lam, v.mean(), np.percentile(v, 5), (v > 0).mean()))
best = max(LAMS, key=lambda l: (round(mins[l], 6), -l))
print('PRE-REGISTERED CHOICE (max-min subgroup): lam', best)
