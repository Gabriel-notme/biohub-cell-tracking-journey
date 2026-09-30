"""Check numpy lgb_np poisson inference vs LightGBM Booster.predict (retrained with identical params/data)."""
import sys, json, pickle, time
import numpy as np
import lightgbm as lgb
sys.path.insert(0, '/workspace/p56stage')
import lgb_np
L = lambda t: pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb'))
TR = [r for t in ['p8_t127a', 'p8_t127b', 'p8_audit32'] for r in L(t)]
TP_REF = float(np.median([r['TP'] for r in TR])); RATIO = float(np.median([r['ntot'] / r['npred'] for r in TR]))
print('train movies', len(TR), 'TP_REF', TP_REF, 'RATIO', RATIO, flush=True)
dep = json.load(open('/workspace/p56stage/jp_lgb.json'))
print('deployed jp_meta', dep['jp_meta'])
nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=nthreads)
X = np.concatenate([r['X'] for r in TR]); y = np.concatenate([r['tp'] for r in TR])
t0 = time.time()
full = lgb.train(params, lgb.Dataset(X, y), 300)
print('trained in %.1fs' % (time.time() - t0), flush=True)
d = full.dump_model(); d['jp_meta'] = {'tp_ref': TP_REF, 'ratio': RATIO}
json.dump(d, open('/workspace/verify/codereview/jp_retrain.json', 'w'))
same_trees = json.dumps(d['tree_info']) == json.dumps(dep['tree_info'])
print('retrained trees identical to deployed jp_lgb.json:', same_trees)
if not same_trees:
    diffs = sum(json.dumps(a) != json.dumps(b) for a, b in zip(d['tree_info'], dep['tree_info']))
    print('  trees differing', diffs, 'of', len(d['tree_info']))
m = lgb_np.load('/workspace/verify/codereview/jp_retrain.json')
print('objective string', repr(m.objective), 'startswith poisson', str(m.objective).startswith('poisson'))
# held-out tables too
H = np.concatenate([r['X'] for t in ['p8_hold36', 'p8_prev4'] for r in L(t)])
for tag, XX in [('train', X), ('hold36+prev4 tables', H)]:
    a = np.exp(m.raw(XX)); b = full.predict(XX); rb = full.predict(XX, raw_score=True); ra = m.raw(XX)
    print('%-20s n=%d max|raw diff| %.3e max rel pred diff %.3e' % (tag, len(XX), np.abs(ra - rb).max(), np.abs(a / b - 1).max()))
# deployed model vs its own numpy port on held-out tables cannot be compared to LightGBM directly unless trees identical
# NaN handling: missing_type None -> LightGBM maps NaN to 0.0
rng = np.random.default_rng(0)
XN = H[:5000].copy()
for j in range(XN.shape[1]):
    XN[rng.random(len(XN)) < 0.1, j] = np.nan
a = m.raw(XN); b = full.predict(XN, raw_score=True)
print('NaN-injected: max|raw diff| %.3e, rows differing(>1e-9) %d / %d' % (np.abs(a - b).max(), int((np.abs(a - b) > 1e-9).sum()), len(XN)))
XZ = np.where(np.isnan(XN), 0.0, XN)
print('NaN->0 numpy vs LightGBM(NaN): max|raw diff| %.3e' % np.abs(m.raw(XZ) - b).max())
XI = H[:2000].copy(); XI[:, 20] = 1e7; XI[:, 1] = 500; XI[:, 2] = 600
print('extrapolated (npred=1e7, t=500): max|raw diff| %.3e' % np.abs(m.raw(XI) - full.predict(XI, raw_score=True)).max())
