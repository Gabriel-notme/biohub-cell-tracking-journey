"""Cross-embryo transfer test of junk pruning: train the Poisson GBM (and tp_ref/ratio) on one embryo's training movies only,
evaluate the exact aggregated change on the OTHER embryo's clean movies (hold36+prev4) and on its training movies."""
import pickle
import numpy as np
import lightgbm as lgb
L = lambda t: pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb'))
TR = [r for t in ['p8_t127a', 'p8_t127b', 'p8_audit32'] for r in L(t)]
CL = [r for t in ['p8_hold36', 'p8_prev4'] for r in L(t)]
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=32)


def agg(rs, lam, tp_ref, ratio):
    n0 = d0 = n1 = d1 = 0.
    for r in rs:
        m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
        nh = r['npred'] * ratio; mh = 1 - 0.1 * (r['npred'] - nh) / nh; be = 0.1 * tp_ref / nh / mh
        sel = r['pred'] < lam * r['len'] * be
        n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum()
        n0 += r['TP'] * m; d0 += w; n1 += (r['TP'] - k) * (m + 0.1 * n / r['ntot']); d1 += w - f
    return n1 / d1 - n0 / d0


for src, dst in [('44b6', '6bba'), ('6bba', '44b6')]:
    tr = [r for r in TR if r['movie'].startswith(src)]
    tp_ref = float(np.median([r['TP'] for r in tr])); ratio = float(np.median([r['ntot'] / r['npred'] for r in tr]))
    m = lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in tr]), np.concatenate([r['tp'] for r in tr])), 300)
    for r in TR + CL: r['pred'] = m.predict(r['X'])
    tgt_clean = [r for r in CL if r['movie'].startswith(dst)]; tgt_train = [r for r in TR if r['movie'].startswith(dst)]
    print('train %s (n=%d, tp_ref %.0f) -> %s clean (n=%d): %s | %s train (n=%d): %s' % (
        src, len(tr), tp_ref, dst, len(tgt_clean), ' '.join('lam%.2f %+.5f' % (l, agg(tgt_clean, l, tp_ref, ratio)) for l in [0.5, 0.75, 1.0, 1.25]),
        dst, len(tgt_train), ' '.join('lam%.2f %+.5f' % (l, agg(tgt_train, l, tp_ref, ratio)) for l in [0.5, 0.75, 1.0, 1.25])))
