"""dv_probe: leave-one-embryo-out feasibility of a learned division-completion classifier on the dv_cand table.
Real rows: P (new GT division) = 1; N, D, X = 0 (all would be official FP forks). U = not evaluable (1% sample, ignored).
Fork rows (synthetic births from existing P15 forks): Ftp=1, Ffp=0, Fnc=weak positive.
Reports per test embryo: AUC, and greedy acceptance (one fork per p and per b, one TP per GT division) at top-k -> TP / FP / precision.
usage: dv_probe.py [extra_json_features_file]"""
import os, sys, json, glob
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_k] = '4'
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

DROP = {'lab', 'src', 'p', 'a', 'b', 'q', 'movie', 'set', 'emb', 'gd', 'y'}


def load(f):
    R = json.load(open(f))
    s, m = os.path.basename(f)[:-5].split('__')
    out = []
    for r in R:
        if r['lab'] == 'U': continue
        r['movie'] = m; r['set'] = s; r['emb'] = m[:4]
        r['gd'] = r['lab'][2:] if r['lab'][:2] in ('P:', 'D:') else ''
        r['lab'] = r['lab'][:1] if r['src'] == 'cand' else r['lab']
        out.append(r)
    return out


def matrix(rows, F):
    return np.array([[(np.nan if r.get(k) is None else r[k]) for k in F] for r in rows], np.float32)


def greedy(rows, sc, ks=(3, 5, 10, 20, 40, 80, 160)):
    o = np.argsort(-sc); used_p, used_b, got = set(), set(), set(); tp = fp = 0; res = {}; hist = []
    for i in o:
        r = rows[i]; key = (r['movie'], r['p']); kb = (r['movie'], r['b'])
        if key in used_p or kb in used_b or (r['movie'], r['a']) in used_b: continue
        used_p.add(key); used_b.add(kb); used_b.add((r['movie'], r['a']))
        if r['typ'] == 1 and r.get('q') is not None: used_p.add((r['movie'], r['q']))
        if r['lab'] == 'P' and (r['movie'], r['gd']) not in got: tp += 1; got.add((r['movie'], r['gd']))
        else: fp += 1
        hist.append((tp, fp, float(sc[i])))
        if len(hist) >= max(ks): break
    for k in ks:
        if k <= len(hist): res[k] = hist[k - 1][:2]
    return res, hist


def run(name, tr, te, F, w_nc=0.0, use_fork=False, params=None):
    trr = [r for r in tr if r['src'] == 'cand']
    y = [int(r['lab'] == 'P') for r in trr]; w = [1.0] * len(trr)
    if use_fork:
        for r in tr:
            if r['src'] != 'fork': continue
            if r['lab'] == 'Ftp': trr.append(r); y.append(1); w.append(1.0)
            elif r['lab'] == 'Ffp': trr.append(r); y.append(0); w.append(1.0)
            elif w_nc > 0: trr.append(r); y.append(1); w.append(w_nc)
    X = matrix(trr, F); y = np.array(y); w = np.array(w)
    pos_w = (y == 0).sum() / max(1, (y == 1).sum() * 20)  # mild rebalancing
    w = w * np.where(y == 1, max(1.0, pos_w), 1.0)
    P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=10.0, verbose=-1, num_threads=8, seed=0)
    if params: P.update(params)
    m = lgb.train(P, lgb.Dataset(X, y, weight=w), num_boost_round=300)
    ter = [r for r in te if r['src'] == 'cand']
    sc = m.predict(matrix(ter, F)); yt = np.array([int(r['lab'] == 'P') for r in ter])
    auc = roc_auc_score(yt, sc) if yt.sum() else float('nan')
    aucs = {}
    for typ in (0, 1):
        ii = [i for i, r in enumerate(ter) if r['typ'] == typ]
        if yt[ii].sum(): aucs[typ] = roc_auc_score(yt[ii], sc[ii])
    res, hist = greedy(ter, sc)
    ngd = len({(r['movie'], r['gd']) for r in ter if r['lab'] == 'P'})
    print('%-34s test %s: nP_div %3d  AUC %.3f (start %.3f stolen %.3f) | greedy top-k TP/FP: %s' % (
        name, ter[0]['emb'], ngd, auc, aucs.get(0, np.nan), aucs.get(1, np.nan), ' '.join('%d:%d/%d' % (k, *v) for k, v in res.items())), flush=True)
    return m, sc, ter


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/nm/dv_cand/*.json'))
    with Pool(24) as p: rows = [r for rs in p.map(load, files) for r in rs]
    extra = {}
    if len(sys.argv) > 1:
        for f in sys.argv[1].split(','):
            for k, v in json.load(open(f)).items(): extra.setdefault(k, {}).update(v)
        for r in rows:
            e = extra.get('%s|%d|%d|%d' % (r['movie'], r['p'], r['a'], r['b']))
            if e: r.update(e)
    c = Counter((r['emb'], r['src'], r['typ'], r['lab']) for r in rows)
    for k in sorted(c): print(k, c[k])
    for e in ['44b6', '6bba']:
        print(e, 'distinct new GT divisions among candidates:', len({(r['movie'], r['gd']) for r in rows if r['emb'] == e and r['lab'] == 'P'}),
              'start:', len({(r['movie'], r['gd']) for r in rows if r['emb'] == e and r['lab'] == 'P' and r['typ'] == 0}),
              'stolen:', len({(r['movie'], r['gd']) for r in rows if r['emb'] == e and r['lab'] == 'P' and r['typ'] == 1}))
    F = sorted({k for r in rows[:5000] for k in r} - DROP)
    F = [k for k in F if isinstance(next((r[k] for r in rows if r.get(k) is not None), 0), (int, float))]
    print('features', len(F), F)
    # univariate AUC (pooled) of each feature, real candidates, per typ
    for typ in (0, 1):
        rr = [r for r in rows if r['src'] == 'cand' and r['typ'] == typ]
        y = np.array([int(r['lab'] == 'P') for r in rr]); line = []
        for k in F:
            x = np.array([(-9 if r.get(k) is None else r[k]) for r in rr], float)
            a = roc_auc_score(y, x); line.append((abs(a - .5), k, a))
        print('typ', typ, 'univariate AUC top:', ' '.join('%s %.2f' % (k, a) for _, k, a in sorted(line, reverse=True)[:14]))
    E = {e: [r for r in rows if r['emb'] == e] for e in ['44b6', '6bba']}
    IMG = [k for k in F if k[:1] in 'ISMCr' and k not in ('starts_near_p',)]
    NOIMG = [k for k in F if k not in IMG]
    B1 = [k for k in F if k.startswith('b1_')]
    for tr_e, te_e in [('6bba', '44b6'), ('44b6', '6bba')]:
        run('struct only', E[tr_e], E[te_e], [k for k in NOIMG if k not in B1])
        run('struct+img', E[tr_e], E[te_e], [k for k in F if k not in B1])
        if B1:
            run('b1 only', E[tr_e], E[te_e], B1)
            run('all (b1+struct+img)', E[tr_e], E[te_e], F)
            run('all + forks(nc w0.3)', E[tr_e], E[te_e], F, w_nc=0.3, use_fork=True)
        run('struct+img + forks(tp/fp)', E[tr_e], E[te_e], [k for k in F if k not in B1], use_fork=True)
        run('struct+img + forks(nc w0.3)', E[tr_e], E[te_e], [k for k in F if k not in B1], w_nc=0.3, use_fork=True)
        run('struct+img + forks(nc w1)', E[tr_e], E[te_e], [k for k in F if k not in B1], w_nc=1.0, use_fork=True)
