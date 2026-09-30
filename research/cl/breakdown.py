"""Breakdown of the relative-rule LOEO result by set (upstream-in-sample t127/audit32 vs clean hold36/prev4) + movie bootstrap;
oracle upside on clean sets; embryo-balanced both-embryo model OOF by embryo."""
import pickle, numpy as np, lightgbm as lgb
R = pickle.load(open('var_preds.pkl', 'rb'))
L = lambda t: pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb'))
X = {}
for t in ['p8_t127a', 'p8_t127b', 'p8_audit32', 'p8_hold36', 'p8_prev4']:
    for r in L(t): X[r['movie']] = r['X']
for r in R: r['X'] = X[r['movie']]


def mg(r, sel):
    m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
    n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum()
    return np.array([r['TP'] * m, w, (r['TP'] - k) * (m + 0.1 * n / r['ntot']), w - f, n, k, r['npred']])


def relsel(r, key, a):
    p = r[key]; return (p / r['len']) < a * p.sum() / r['len'].sum()


def g(P): return P[:, 2].sum() / P[:, 3].sum() - P[:, 0].sum() / P[:, 1].sum()


def boot(P, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.array([g(P[rng.integers(0, len(P), len(P))]) for _ in range(B)])
    return np.percentile(v, 5), np.percentile(v, 95), (v > 0).mean()


AL = [0.03, 0.05, 0.1]
print('== LOEO 44b6->6bba (FULL feats, relative rule) by set')
for s in ['p8_t127a', 'p8_t127b', 'p8_audit32', 'p8_hold36', 'p8_prev4', 'CLEAN']:
    rs = [r for r in R if r['movie'].startswith('6bba') and (r['set'] == s or (s == 'CLEAN' and r['set'] in ('p8_hold36', 'p8_prev4')))]
    line = '%-10s n=%3d' % (s, len(rs))
    for a in AL:
        P = np.array([mg(r, relsel(r, 'x_FULL', a)) for r in rs]); lo, hi, pp = boot(P)
        line += ' | a%.2f %+.5f [5%% %+.5f 95%% %+.5f P>0 %.2f] rmfrac %.3f' % (a, g(P), lo, hi, pp, P[:, 4].sum() / P[:, 6].sum())
    print(line)
print('== in-distribution full model (both embryos, trained on t127+audit32) on CLEAN40, relative rule, with bootstrap')
cl = [r for r in R if r['set'] in ('p8_hold36', 'p8_prev4')]
for a in [0.03, 0.05, 0.075, 0.1]:
    P = np.array([mg(r, relsel(r, 'o_FULL', a)) for r in cl]); lo, hi, pp = boot(P)
    print('  a%.3f %+.5f [5%% %+.5f 95%% %+.5f P>0 %.2f]' % (a, g(P), lo, hi, pp))
print('== oracle upside (remove exactly tp==0 tracks) and oracle restricted to len<=10')
for nm, rs in [('clean40', cl), ('clean 6bba', [r for r in cl if r['movie'].startswith('6bba')]), ('clean 44b6', [r for r in cl if r['movie'].startswith('44b6')])]:
    P = np.array([mg(r, r['tp'] == 0) for r in rs]); P2 = np.array([mg(r, (r['tp'] == 0) & (r['len'] <= 10)) for r in rs])
    print('  %-11s oracle %+.5f (rm %.3f of nodes) | oracle len<=10 %+.5f' % (nm, g(P), P[:, 4].sum() / P[:, 6].sum(), g(P2)))
# embryo-balanced both-embryo model: does its OOF gain come from both embryos? (train OOF by embryo)
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=16)
TR = [r for r in R if r['set'] not in ('p8_hold36', 'p8_prev4')]
print('== train OOF (both embryos) by embryo, relative rule (o_FULL = unweighted)')
for e in ['44b6', '6bba']:
    rs = [r for r in TR if r['movie'].startswith(e)]
    print('  %s n=%d ' % (e, len(rs)) + ' '.join('a%.2f %+.5f' % (a, g(np.array([mg(r, relsel(r, 'o_FULL', a)) for r in rs]))) for a in [0.03, 0.05, 0.1, 0.15]))
