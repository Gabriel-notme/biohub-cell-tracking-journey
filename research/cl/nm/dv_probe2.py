"""dv_probe2: start-type only (daughter-birth formulation). LOEO: train on embryo A (real start candidates and/or synthetic births
from existing forks), test on embryo B real start candidates. Prints AUC, rank of every positive among test negatives, greedy top-k.
usage: dv_probe2.py [extra_features_json]"""
import os, sys, json, glob
from collections import Counter
from multiprocessing import Pool
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/workspace/cl/nm')
from dv_probe import load, matrix, greedy, DROP


def loads(f):
    return [r for r in load(f) if r['typ'] == 0]


def fit(tr, F, mode, w_nc=0.3, seed=0):
    rr, y, w = [], [], []
    for r in tr:
        if r['src'] == 'cand' and mode in ('real', 'real+fork', 'real+fork+nc'):
            rr.append(r); y.append(int(r['lab'] == 'P')); w.append(1.0)
        if r['src'] == 'fork' and mode != 'real':
            if r['lab'] == 'Ftp': rr.append(r); y.append(1); w.append(1.0)
            elif r['lab'] == 'Ffp': rr.append(r); y.append(0); w.append(1.0)
            elif 'nc' in mode: rr.append(r); y.append(1); w.append(w_nc)
        if r['src'] == 'cand' and mode in ('fork', 'fork+nc'):  # negatives still come from real candidates
            if r['lab'] != 'P': rr.append(r); y.append(0); w.append(1.0)
    y = np.array(y); w = np.array(w)
    P = dict(objective='binary', learning_rate=0.03, num_leaves=7, min_data_in_leaf=30, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=10.0, verbose=-1, num_threads=8, seed=seed)
    return lgb.train(P, lgb.Dataset(matrix(rr, F), y, weight=w), num_boost_round=250), int(y.sum())


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/nm/dv_cand/*.json'))
    with Pool(24) as p: rows = [r for rs in p.map(loads, files) for r in rs]
    if len(sys.argv) > 1:
        extra = {}
        for f in sys.argv[1].split(','):
            for k, v in json.load(open(f)).items(): extra.setdefault(k, {}).update(v)
        miss = 0
        for r in rows:
            e = extra.get('%s|%d|%d|%d' % (r['movie'], r['p'], r['a'], r['b']))
            if e: r.update(e)
            else: miss += 1
        print('extra features missing for', miss, 'of', len(rows))
    F = sorted({k for r in rows[:3000] for k in r} - DROP)
    F = [k for k in F if isinstance(next((r[k] for r in rows if r.get(k) is not None), 0), (int, float)) and not k.startswith('q') and k not in ('d_qp', 'd_qb', 'Iq', 'Cq', 'typ')]
    IMG = [k for k in F if k[:1] in 'ISMCr']
    B1 = [k for k in F if k.startswith('b1_')]
    E = {e: [r for r in rows if r['emb'] == e] for e in ['44b6', '6bba']}
    FS = {'struct': [k for k in F if k not in IMG and k not in B1], 'struct+img': [k for k in F if k not in B1]}
    if B1: FS['b1'] = B1; FS['b1+struct'] = [k for k in F if k not in IMG]; FS['all'] = F
    for tr_e, te_e in [('6bba', '44b6'), ('44b6', '6bba')]:
        te = [r for r in E[te_e] if r['src'] == 'cand']; yt = np.array([int(r['lab'] == 'P') for r in te])
        print('== test', te_e, 'real start candidates', len(te), 'P rows', yt.sum(), 'distinct', len({(r['movie'], r['gd']) for r in te if r['lab'] == 'P'}))
        for fs, FF in FS.items():
            for mode in ['real', 'fork', 'real+fork', 'real+fork+nc', 'fork+nc']:
                m, npos = fit(E[tr_e], FF, mode)
                sc = m.predict(matrix(te, FF))
                auc = roc_auc_score(yt, sc)
                neg = np.sort(sc[yt == 0])[::-1]
                ranks = sorted(int((neg > s).sum()) for s in sc[yt == 1])
                res, hist = greedy(te, sc, ks=(2, 4, 8, 16, 32, 64))
                print('  %-11s %-13s npos_tr %4d AUC %.3f | neg above each P: %s | greedy %s' % (fs, mode, npos, auc, ranks[:12], ' '.join('%d:%d/%d' % (k, *v) for k, v in res.items())), flush=True)
