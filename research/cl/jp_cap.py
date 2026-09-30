"""Junk prune with a per-movie cap on the removed-node fraction (lowest predicted TP-per-node tracks first).
Train OOF (same folds as jp_robust) + subgroup min; also the cross-embryo stress test (train 44b6 -> 6bba) with the cap."""
import pickle
import numpy as np
import lightgbm as lgb
L = lambda t: pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb'))
TR = []
for t in ['p8_t127a', 'p8_t127b', 'p8_audit32']:
    for r in L(t): r['set'] = t; TR.append(r)
CL = [r for t in ['p8_hold36', 'p8_prev4'] for r in L(t)]
TP_REF = float(np.median([r['TP'] for r in TR])); RATIO = float(np.median([r['ntot'] / r['npred'] for r in TR]))
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=32)
rng = np.random.default_rng(0); idx = np.arange(len(TR)); rng.shuffle(idx); fold = {int(i): k % 5 for k, i in enumerate(idx)}
for k in range(5):
    a = [r for i, r in enumerate(TR) if fold[i] != k]; b = [r for i, r in enumerate(TR) if fold[i] == k]
    m = lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in a]), np.concatenate([r['tp'] for r in a])), 300)
    for r in b: r['oof'] = m.predict(r['X'])


def select(r, key, lam, cap, tp_ref, ratio):
    nh = r['npred'] * ratio; mh = 1 - 0.1 * (r['npred'] - nh) / nh; be = 0.1 * tp_ref / nh / mh
    sel = r[key] < lam * r['len'] * be
    if cap is not None and r['len'][sel].sum() > cap * r['npred']:
        idx = np.flatnonzero(sel); order = idx[np.argsort(r[key][idx] / r['len'][idx])]
        cum = np.cumsum(r['len'][order]); keep = order[cum <= cap * r['npred']]
        sel = np.zeros_like(sel); sel[keep] = True
    return sel


def agg(rs, key, lam, cap, tp_ref=TP_REF, ratio=RATIO):
    n0 = d0 = n1 = d1 = 0.; fr = []
    for r in rs:
        m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
        sel = select(r, key, lam, cap, tp_ref, ratio)
        n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum(); fr.append(n / r['npred'])
        n0 += r['TP'] * m; d0 += w; n1 += (r['TP'] - k) * (m + 0.1 * n / r['ntot']); d1 += w - f
    return n1 / d1 - n0 / d0, np.max(fr), np.percentile(fr, 90)


groups = {}
for r in TR: groups.setdefault((r['set'], r['movie'][:4]), []).append(r)
for cap in [None, 0.10, 0.05, 0.03, 0.02]:
    g, mx, p90 = agg(TR, 'oof', 1.0, cap)
    mn = min(agg(v, 'oof', 1.0, cap)[0] for v in groups.values())
    print('cap %-5s train OOF lam1.0 %+.5f  subgroup min %+.5f  removed frac max %.3f p90 %.3f' % (cap, g, mn, mx, p90))
# stress test: model trained on 44b6 only, applied to 6bba (catastrophic without cap)
tr = [r for r in TR if r['movie'].startswith('44b6')]
tp_ref = float(np.median([r['TP'] for r in tr])); ratio = float(np.median([r['ntot'] / r['npred'] for r in tr]))
m = lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in tr]), np.concatenate([r['tp'] for r in tr])), 300)
for r in TR + CL: r['x44'] = m.predict(r['X'])
for cap in [None, 0.10, 0.05, 0.03, 0.02]:
    print('cap %-5s STRESS 44b6->6bba clean %+.5f  6bba train %+.5f' % (cap, agg([r for r in CL if r['movie'].startswith('6bba')], 'x44', 1.0, cap, tp_ref, ratio)[0],
                                                                         agg([r for r in TR if r['movie'].startswith('6bba')], 'x44', 1.0, cap, tp_ref, ratio)[0]))
full = lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in TR]), np.concatenate([r['tp'] for r in TR])), 300)
for r in CL: r['full'] = full.predict(r['X'])
for cap in [None, 0.10, 0.05, 0.03, 0.02]:
    for s in ['hold36', 'prev4']:
        pass
print('clean sets (full model, for reference only; cap chosen on train): ' + ' '.join('cap %s %+.5f' % (c, agg(CL, 'full', 1.0, c)[0]) for c in [None, 0.10, 0.05, 0.03, 0.02]))
