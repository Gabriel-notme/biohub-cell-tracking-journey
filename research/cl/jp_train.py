"""Train the junk-prune Poisson GBM on train tables, OOF by movie; exact aggregated adj-J change for a lam grid (train OOF, test sets).
usage: jp_train.py <train tags,> <test tags,> <out.json>"""
import sys, json, pickle
import numpy as np
import lightgbm as lgb
TRAIN = sys.argv[1].split(','); TEST = sys.argv[2].split(','); OUT = sys.argv[3]
L = lambda t: pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb'))
TR = [r for t in TRAIN for r in L(t)]
TP_REF = float(np.median([r['TP'] for r in TR])); RATIO = float(np.median([r['ntot'] / r['npred'] for r in TR]))
print('train movies', len(TR), 'TP_REF %.0f RATIO %.4f' % (TP_REF, RATIO))
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=32)
NR = 300
rng = np.random.default_rng(0); idx = np.arange(len(TR)); rng.shuffle(idx); fold = {int(i): k % 5 for k, i in enumerate(idx)}
for k in range(5):
    a = [r for i, r in enumerate(TR) if fold[i] != k]; b = [r for i, r in enumerate(TR) if fold[i] == k]
    m = lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in a]), np.concatenate([r['tp'] for r in a])), NR)
    for r in b: r['pred'] = m.predict(r['X']) if len(r['X']) else np.zeros(0)
full = lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in TR]), np.concatenate([r['tp'] for r in TR])), NR)
d = full.dump_model(); d['jp_meta'] = {'tp_ref': TP_REF, 'ratio': RATIO}
json.dump(d, open(OUT, 'w'))


def agg(rs, lam, key='pred'):
    num0 = den0 = num1 = den1 = 0.; nn = kk = ff = 0
    for r in rs:
        m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
        num0 += r['TP'] * m; den0 += w
        ntot_hat = r['npred'] * RATIO; m_hat = 1 - 0.1 * (r['npred'] - ntot_hat) / ntot_hat; be = 0.1 * TP_REF / ntot_hat / m_hat
        sel = r[key] < lam * r['len'] * be if len(r['len']) else np.zeros(0, bool)
        n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum()
        num1 += (r['TP'] - k) * (m + 0.1 * n / r['ntot']); den1 += w - f; nn += n; kk += k; ff += f
    return num1 / den1 - num0 / den0, nn, kk, ff


LAMS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
print('train OOF:', ' '.join('lam%.2f %+.5f(n%d tp%d fp%d)' % ((l,) + agg(TR, l)) for l in LAMS))
for t in TEST:
    rs = L(t)
    for r in rs: r['pred'] = full.predict(r['X']) if len(r['X']) else np.zeros(0)
    print('%-12s' % t, ' '.join('lam%.2f %+.5f(n%d tp%d fp%d)' % ((l,) + agg(rs, l)) for l in LAMS))
    # oracle for reference: remove exactly tracks with tp==0
    print('%-12s oracle(tp==0) %+.5f' % (t, agg([dict(r, pred=np.where(r['tp'] > 0, 1e9, -1.)) for r in rs], 0.0001)[0]))
