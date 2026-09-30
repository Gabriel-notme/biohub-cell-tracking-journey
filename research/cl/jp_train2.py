"""Compare junk-prune models with / without image features on TRAIN OOF only (movie-grouped 5 fold), per subgroup min and bootstrap.
usage: jp_train2.py"""
import pickle
import numpy as np
import lightgbm as lgb
TRAIN = ['p8_t127a', 'p8_t127b', 'p8_audit32']
TR = []
for t in TRAIN:
    img = pickle.load(open('/workspace/cl/jp/%s_img.pkl' % t, 'rb'))
    for r in pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb')):
        r['set'] = t; r['IMG'] = img[r['movie']]; assert len(r['IMG']) == len(r['X']); TR.append(r)
TP_REF = float(np.median([r['TP'] for r in TR])); RATIO = float(np.median([r['ntot'] / r['npred'] for r in TR]))
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=32)
rng = np.random.default_rng(0); idx = np.arange(len(TR)); rng.shuffle(idx); fold = {int(i): k % 5 for k, i in enumerate(idx)}


def feat(r, use_img):
    return np.column_stack([r['X'], r['IMG']]) if use_img else r['X']


def parts(r, lam, key):
    m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
    ntot_hat = r['npred'] * RATIO; m_hat = 1 - 0.1 * (r['npred'] - ntot_hat) / ntot_hat; be = 0.1 * TP_REF / ntot_hat / m_hat
    sel = r[key] < lam * r['len'] * be if len(r['len']) else np.zeros(0, bool)
    n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum()
    return r['TP'] * m, w, (r['TP'] - k) * (m + 0.1 * n / r['ntot']), w - f


def gain(rs, lam, key):
    P = np.array([parts(r, lam, key) for r in rs]); return P[:, 2].sum() / P[:, 3].sum() - P[:, 0].sum() / P[:, 1].sum()


def auc(p, yy):
    o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(len(p)); n1 = yy.sum(); n0 = len(yy) - n1
    return (rk[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


for use_img, key in [(False, 'o0'), (True, 'o1')]:
    for k in range(5):
        a = [r for i, r in enumerate(TR) if fold[i] != k]; b = [r for i, r in enumerate(TR) if fold[i] == k]
        m = lgb.train(params, lgb.Dataset(np.concatenate([feat(r, use_img) for r in a]), np.concatenate([r['tp'] for r in a])), 300)
        for r in b: r[key] = m.predict(feat(r, use_img))
    p = np.concatenate([r[key] / r['len'] for r in TR]); y = np.concatenate([(r['tp'] > 0).astype(int) for r in TR])
    groups = {}
    for r in TR: groups.setdefault((r['set'], r['movie'][:4]), []).append(r)
    line = 'img=%s AUC(tp>0) %.4f |' % (use_img, auc(p, y))
    for lam in [0.75, 1.0, 1.25, 1.5]:
        g = [gain(v, lam, key) for v in groups.values()]
        line += ' lam%.2f all %+.5f min %+.5f;' % (lam, gain(TR, lam, key), min(g))
    print(line, flush=True)
