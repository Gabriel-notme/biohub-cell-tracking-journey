"""Density-invariant junk pruning: remove a fork-free isolated track k iff its predicted annotated-edge rate r_k = E[TP_k]/len_k
is below alpha x the movie's node-weighted mean predicted rate. Derivation: removing k pays iff r_k < 0.1*TP_movie/(N_total*m);
with TP_movie ~ rbar*N_pred the embryo-level annotation density cancels -> alpha ~ 0.1*N_pred/(N_total*m) ~ 0.096.
Tests: train OOF (both embryos), and leave-one-embryo-out in both directions (the stress test that broke the absolute rule)."""
import pickle
import numpy as np
import lightgbm as lgb
L = lambda t: pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb'))
TR = []
for t in ['p8_t127a', 'p8_t127b', 'p8_audit32']:
    for r in L(t): r['set'] = t; TR.append(r)
CL = []
for t in ['p8_hold36', 'p8_prev4']:
    for r in L(t): r['set'] = t; CL.append(r)
ALL = TR + CL
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=32)


def fit(rs):
    return lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in rs]), np.concatenate([r['tp'] for r in rs])), 300)


def agg(rs, key, alpha):
    n0 = d0 = n1 = d1 = 0.; nn = kk = 0
    for r in rs:
        m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
        p = r[key]; rate = p / r['len']; rbar = p.sum() / max(1, r['len'].sum())
        sel = rate < alpha * rbar
        n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum(); nn += n; kk += k
        n0 += r['TP'] * m; d0 += w; n1 += (r['TP'] - k) * (m + 0.1 * n / r['ntot']); d1 += w - f
    return n1 / d1 - n0 / d0, nn, kk


AL = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
# 1) train OOF (movie-grouped, both embryos) -- same folds as before
rng = np.random.default_rng(0); idx = np.arange(len(TR)); rng.shuffle(idx); fold = {int(i): k % 5 for k, i in enumerate(idx)}
for k in range(5):
    m = fit([r for i, r in enumerate(TR) if fold[i] != k])
    for i, r in enumerate(TR):
        if fold[i] == k: r['oof'] = m.predict(r['X'])
print('train OOF      ', ' '.join('a%.2f %+.5f(n%d tp%d)' % ((a,) + agg(TR, 'oof', a)) for a in AL))
groups = {}
for r in TR: groups.setdefault((r['set'], r['movie'][:4]), []).append(r)
print('train OOF min subgroup', ' '.join('a%.2f %+.5f' % (a, min(agg(v, 'oof', a)[0] for v in groups.values())) for a in AL))
# 2) leave-one-embryo-out: train on all movies (all sets) of embryo A, evaluate on every movie of embryo B
for src, dst in [('44b6', '6bba'), ('6bba', '44b6')]:
    m = fit([r for r in ALL if r['movie'].startswith(src)])
    tgt = [r for r in ALL if r['movie'].startswith(dst)]
    for r in tgt: r['x'] = m.predict(r['X'])
    print('LOEO %s->%s all %d' % (src, dst, len(tgt)), ' '.join('a%.2f %+.5f(n%d tp%d)' % ((a,) + agg(tgt, 'x', a)) for a in AL))
    tc = [r for r in tgt if r['set'] in ('p8_hold36', 'p8_prev4')]
    print('   of which clean %d' % len(tc), ' '.join('a%.2f %+.5f' % (a, agg(tc, 'x', a)[0]) for a in AL))
