"""LOEO evaluation of a learned link re-scorer on P15 residual candidates (rl_cands.py rows, evaluable only).
gain(row) = dTP - J*dFP of the action (add s->d, drop s->c, drop q->d).  Target y = gain > 0.
Train on one embryo (all sets), choose the threshold on the training embryo's out-of-fold predictions (movie-grouped 4-fold),
apply greedily on the other embryo (each node touched once), convert exact count deltas to official score deltas.
usage: rl_train.py [extra_feature_pkl_dir]"""
import os, sys, json, glob, pickle
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/cl/nm')
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict, Counter
import numpy as np
import lightgbm as lgb
from tracking_cellmot.metrics import summarise
from rl_cands import FEATS
EXTRA = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != '-' else None
DROP = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else set()
base = {}
for f in glob.glob('/workspace/cl/nm/rl_oracle/*.json'):
    d = json.load(open(f)); base[d['movie']] = dict(d['scores']['base'], set=d['set'])
M = []
for f in sorted(glob.glob('/workspace/cl/nm/rl_cands/*.pkl')):
    s, m = os.path.basename(f)[:-4].split('__'); z = pickle.load(open(f, 'rb'))
    X = z['X']; names = list(FEATS)
    if EXTRA:
        e = pickle.load(open(os.path.join(EXTRA, os.path.basename(f)), 'rb')); X = np.column_stack([X, e['X']]); names += e['names']
    M.append(dict(set=s, movie=m, emb=m[:4], X=X, L=z['L'], ntot=z['ntot']))
keep = [i for i, n in enumerate(names) if n not in DROP]; names = [names[i] for i in keep]
for mv in M: mv['X'] = mv['X'][:, keep]
J = 0.97
for mv in M:
    L = mv['L']
    mv['dtp'] = L['tp_sd'] - L['tp_sc'] - L['tp_qd']
    mv['dfp'] = (L['v_sd'] - L['tp_sd']) - (L['v_sc'] - L['tp_sc']) - (L['v_qd'] - L['tp_qd'])
    mv['y'] = ((L['tp_sd'] == 1) & (L['tp_sc'] == 0) & (L['tp_qd'] == 0)).astype(int)  # the proposed link itself is correct
DOM = os.environ.get('DOMAIN', 'all')
if DOM != 'all':
    fi = {n: i for i, n in enumerate(names)}
    for mv in M:
        X = mv['X']; typ = X[:, fi['typ']]; fe = X[:, fi['fe_sd']]
        k = {'pre': fe >= 0, 'ff': typ == 0, 'nosw': typ <= 2, 'pre_nosw': (fe >= 0) & (typ <= 2)}[DOM]
        mv['X'] = X[k]; mv['dtp'] = mv['dtp'][k]; mv['dfp'] = mv['dfp'][k]; mv['y'] = mv['y'][k]; mv['L'] = {a: b[k] for a, b in mv['L'].items()}
    print('DOMAIN', DOM)
print('features', len(names), 'movies', len(M), 'rows', sum(len(m['y']) for m in M), 'pos', sum(m['y'].sum() for m in M), 'tp_sd', sum(m['L']['tp_sd'].sum() for m in M))
for e in ['44b6', '6bba']:
    mm = [m for m in M if m['emb'] == e]
    print('  %s rows %d pos %d (tp_sd %d) cand total %d' % (e, sum(len(m['y']) for m in mm), sum(m['y'].sum() for m in mm), sum(m['L']['tp_sd'].sum() for m in mm), sum(m['ntot'] for m in mm)))
P = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, seed=0, num_threads=8)
NR = 500


def fit(ms):
    X = np.concatenate([m['X'] for m in ms]); y = np.concatenate([m['y'] for m in ms])
    return lgb.train(P, lgb.Dataset(X, y), NR)


def greedy(m, p, th):
    L = m['L']; o = np.argsort(-p); used = set(); tp = fp = n = 0
    for i in o:
        if p[i] < th: break
        ks = [L['s'][i], L['d'][i]] + [x for x in (L['c'][i], L['q'][i]) if x >= 0]
        if any(k in used for k in ks): continue
        used.update(ks); tp += m['dtp'][i]; fp += m['dfp'][i]; n += 1
    return tp, fp, n


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p)); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


def score_delta(ms, res):
    a = summarise([base[m['movie']] for m in ms]); rows = []
    for m in ms:
        b = dict(base[m['movie']]); dtp, dfp, _ = res[m['movie']]
        b['edge_tp'] += int(dtp); b['edge_fp'] += int(dfp); b['edge_fn'] -= int(dtp)
        W = b['edge_tp'] + b['edge_fp'] + b['edge_fn']; jj = b['edge_tp'] / W
        b['adj_edge_jaccard'] = max(0., jj * (1 - 0.1 * b['total_node_ratio'])); rows.append(b)
    return summarise(rows)['score'] - a['score']


THS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CL = ('hold36', 'prev4')
if os.environ.get('MODE', 'loeo') == 'loeo':
    SPL = [('44b6', '6bba', [m for m in M if m['emb'] == '44b6'], [m for m in M if m['emb'] == '6bba']), ('6bba', '44b6', [m for m in M if m['emb'] == '6bba'], [m for m in M if m['emb'] == '44b6'])]
else:  # deploy-like: train on the link-model training sets (both embryos), test on clean40 (same embryos, unseen movies)
    tr_ = [m for m in M if m['set'] not in CL]
    SPL = [('insample', 'clean40_' + e, tr_, [m for m in M if m['set'] in CL and m['emb'] == e]) for e in ['44b6', '6bba']] + [('insample', 'clean40', tr_, [m for m in M if m['set'] in CL])]
for tr, te, mtr, mte in SPL:
    # out-of-fold on the training embryo to pick a threshold
    rng = np.random.default_rng(0); idx = rng.permutation(len(mtr)); fold = {mtr[i]['movie']: k % 4 for k, i in enumerate(idx)}
    oof = {}
    for k in range(4):
        b = fit([m for m in mtr if fold[m['movie']] != k])
        for m in mtr:
            if fold[m['movie']] == k: oof[m['movie']] = b.predict(m['X'])
    best = (0., None)
    for th in THS:
        res = {m['movie']: greedy(m, oof[m['movie']], th) for m in mtr}; d = score_delta(mtr, res)
        if d > best[0]: best = (d, th)
    b = fit(mtr)
    pte = {m['movie']: b.predict(m['X']) for m in mte}
    yy = np.concatenate([m['L']['tp_sd'] for m in mte]); pp = np.concatenate([pte[m['movie']] for m in mte]); yg = np.concatenate([m['y'] for m in mte])
    print('\n== train %s -> test %s | test AUC(tp_sd) %.3f AUC(gain>0) %.3f | train-OOF best th %s (%+.5f)' % (tr, te, auc(pp, yy), auc(pp, yg), best[1], best[0]))
    for th in THS:
        res = {m['movie']: greedy(m, pte[m['movie']], th) for m in mte}
        n = sum(r[2] for r in res.values()); dtp = sum(r[0] for r in res.values()); dfp = sum(r[1] for r in res.values())
        cl = [m for m in mte if m['set'] in ('hold36', 'prev4')]
        up = sum(1 for m in mte if res[m['movie']][0] - J * res[m['movie']][1] > 0); dn = sum(1 for m in mte if res[m['movie']][0] - J * res[m['movie']][1] < 0)
        print('  th %.2f%s applied %5d dTP %+5d dFP %+5d | score %+.5f | clean %+.5f | movies up %d down %d' % (th, '*' if th == best[1] else ' ', n, dtp, dfp, score_delta(mte, res), score_delta(cl, res), up, dn))
    # oracle on the same rows
    res = {m['movie']: greedy(m, m['y'].astype(float) + 1e-3 * m['dtp'], 0.5) for m in mte}
    print('  oracle(rows) dTP %+d dFP %+d score %+.5f' % (sum(r[0] for r in res.values()), sum(r[1] for r in res.values()), score_delta(mte, res)))
    imp = sorted(zip(b.feature_importance('gain'), names), reverse=True)[:12]
    print('  top feats', [(f, int(g)) for g, f in imp])
