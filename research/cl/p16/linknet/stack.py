"""Stacked LOEO LightGBM on rl_cands rows: geometry (+b1) (+linknet CNN pair logits).
Training embryo E uses OUT-OF-FOLD CNN scores (out/oof_E_{0,1}), the held-out embryo uses the CNN trained on all of E (out/v1_E).
Threshold chosen on E by a movie-grouped 4-fold CV of the LightGBM (greedy one-touch, exact count -> official score conversion).
Writes decision pickles for the chosen threshold: out/dec_<variant>.pkl {movie: [(s_id, d_id, c_id, q_id)]}.
usage: stack.py [cnn_tag_prefix=v1] [oof_prefix=oof]"""
import os, sys, glob, json, pickle
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/cl/nm')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, lightgbm as lgb
from tracking_cellmot.metrics import summarise
from rl_cands import FEATS
TAG = sys.argv[1] if len(sys.argv) > 1 else 'v1'
OOF = sys.argv[2] if len(sys.argv) > 2 else 'oof'
LN = '/workspace/cl/p16/linknet'
CL = ('hold36', 'prev4')
base = {}; FLIP = {}
for f in glob.glob('/workspace/cl/nm/rl_oracle/*.json'):
    d = json.load(open(f)); base[d['movie']] = d['scores']['base']; FLIP[d['movie']] = {(x['p1'], x['p2']): x['flip'] for x in d['fixes']}


def cnn_feats(tb, npz):
    z = np.load(npz); P = z['P']; lg = z['lg']
    key = P[:, 0] * 10 ** 7 + P[:, 1]; o = np.argsort(key); key = key[o]; lg = lg[o]

    def look(a, b):
        k = np.maximum(a, 0) * 10 ** 7 + np.maximum(b, 0); j = np.clip(np.searchsorted(key, k), 0, len(key) - 1)
        return np.where((a >= 0) & (b >= 0) & (key[j] == k), lg[j], np.nan)
    r = tb['rows']; S_, D_, C_, Q_ = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    asd, asc, aqd, aqc = look(S_, D_), look(S_, C_), look(Q_, D_), look(Q_, C_)
    ls = lambda a: -np.logaddexp(0, -a); l1 = lambda a: -np.logaddexp(0, a)
    A1 = ls(asd) + np.where(np.isnan(asc), 0, l1(np.nan_to_num(asc))) + np.where(np.isnan(aqd), 0, l1(np.nan_to_num(aqd)))
    A2 = asd - np.fmax(np.nan_to_num(asc, nan=-30), np.nan_to_num(aqd, nan=-30))
    sw = np.where(np.isnan(aqc), -30, asd + np.nan_to_num(aqc) - np.nan_to_num(asc) - np.nan_to_num(aqd))
    return np.column_stack([asd, np.nan_to_num(asc, nan=-30), np.nan_to_num(aqd, nan=-30), np.nan_to_num(aqc, nan=-30), A1, A2, sw]).astype(np.float32)


M = []
for f in sorted(glob.glob(LN + '/tab/*.pkl')):
    s, m = os.path.basename(f)[:-4].split('__'); tb = pickle.load(open(f, 'rb'))
    if len(tb['rows']) == 0: continue
    emb = m[:4]; other = '6bba' if emb == '44b6' else '44b6'
    z = pickle.load(open('/workspace/cl/nm/rl_cands/%s__%s.pkl' % (s, m), 'rb')); b1 = pickle.load(open('/workspace/cl/nm/rl_b1f/%s__%s.pkl' % (s, m), 'rb'))['X']
    # CNN features as TEST embryo (model trained on the other embryo) and as TRAINING embryo (OOF within own embryo)
    cte = cnn_feats(tb, '%s/out/%s_%s/scores/%s__%s.npz' % (LN, TAG, other, s, m))
    fo = [p for p in glob.glob('%s/out/%s_%s_*/scores/%s__%s.npz' % (LN, OOF, emb, s, m))]
    assert len(fo) <= 1, (m, fo)
    ctr = cnn_feats(tb, fo[0]) if fo else None
    ids = tb['ids']
    fl = np.array([FLIP[m].get((int(ids[a]), int(ids[b])), 0) for a, b in zip(tb['rows'][:, 0], tb['rows'][:, 1])], int) * tb['y'].astype(int)
    M.append(dict(set=s, movie=m, emb=emb, G=z['X'], B=b1, Cte=cte, Ctr=ctr, y=tb['y'].astype(int), dtp=tb['dtp'].astype(int), dfp=tb['dfp'].astype(int),
                  rows=tb['rows'], ids=ids, flip=fl))
print('movies', len(M), flush=True)
P = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, seed=0, num_threads=16)
NR = 500
THS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
VARS = {'geo': lambda m, tr: m['G'], 'geob1': lambda m, tr: np.column_stack([m['G'], m['B']]),
        'geocnn': lambda m, tr: np.column_stack([m['G'], m['Ctr'] if tr else m['Cte']]),
        'geob1cnn': lambda m, tr: np.column_stack([m['G'], m['B'], m['Ctr'] if tr else m['Cte']]),
        'cnn': lambda m, tr: (m['Ctr'] if tr else m['Cte'])}
if len(sys.argv) > 3: VARS = {k: VARS[k] for k in sys.argv[3].split(',')}


def auc(p, y):
    o = np.argsort(p, kind='mergesort'); rk = np.empty(len(p)); rk[o] = np.arange(len(p)); n1 = y.sum(); n0 = len(y) - n1
    return (rk[y == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


def score_delta(res):
    ms = list(res)
    if not ms: return 0.
    a = summarise([base[m] for m in ms]); out = []
    for m in ms:
        b = dict(base[m]); dtp, dfp = res[m][:2]
        b['edge_tp'] += int(dtp); b['edge_fp'] += int(dfp); b['edge_fn'] -= int(dtp)
        W = b['edge_tp'] + b['edge_fp'] + b['edge_fn']; b['adj_edge_jaccard'] = max(0., b['edge_tp'] / W * (1 - 0.1 * b['total_node_ratio'])); out.append(b)
    return summarise(out)['score'] - a['score']


def greedy(m, p, th):
    o = np.argsort(-p); used = set(); tp = fp = n = ny = nf = 0; dec = []
    for i in o:
        if p[i] < th: break
        ks = [x for x in m['rows'][i] if x >= 0]
        if any(k in used for k in ks): continue
        used.update(ks); tp += m['dtp'][i]; fp += m['dfp'][i]; n += 1; ny += m['y'][i]; nf += m['flip'][i]
        dec.append(tuple(int(m['ids'][x]) if x >= 0 else -1 for x in m['rows'][i]))
    return tp, fp, n, ny, nf, dec


def topk(ms, pr, Ks=(30, 50, 100)):
    allv = np.concatenate([pr[m['movie']] for m in ms]); mix = np.concatenate([np.full(len(m['y']), i) for i, m in enumerate(ms)])
    rix = np.concatenate([np.arange(len(m['y'])) for m in ms]); o = np.argsort(-allv, kind='mergesort')
    used = [set() for _ in ms]; res = {}; n = ny = 0; out = []
    for i in o:
        m = ms[mix[i]]; j = rix[i]; ks = [x for x in m['rows'][j] if x >= 0]
        if any(k in used[mix[i]] for k in ks): continue
        used[mix[i]].update(ks); n += 1; ny += m['y'][j]; r = res.setdefault(m['movie'], [0, 0]); r[0] += m['dtp'][j]; r[1] += m['dfp'][j]
        if n in Ks: out.append('K%d %.2f (p>=%.2f) %+.5f' % (n, ny / n, allv[i], score_delta(res)))
        if n >= max(Ks): break
    return ' | '.join(out)


fit = lambda X, y: lgb.train(P, lgb.Dataset(X, y), NR)
DEC = {}
DIRS = os.environ.get('DIRS', '44b6,6bba').split(',')
for tr, te in [('44b6', '6bba'), ('6bba', '44b6')]:
    if tr not in DIRS: continue
    mtr = [m for m in M if m['emb'] == tr]; mte = [m for m in M if m['emb'] == te]
    rng = np.random.default_rng(0); idx = rng.permutation(len(mtr)); fold = {mtr[i]['movie']: k % 4 for k, i in enumerate(idx)}
    for vn, fx in VARS.items():
        oof = {}
        for k in range(4):
            b = fit(np.concatenate([fx(m, True) for m in mtr if fold[m['movie']] != k]), np.concatenate([m['y'] for m in mtr if fold[m['movie']] != k]))
            for m in mtr:
                if fold[m['movie']] == k: oof[m['movie']] = b.predict(fx(m, True))
        best = (0., None)
        for th in THS:
            res = {}
            for m in mtr:
                r = greedy(m, oof[m['movie']], th); res[m['movie']] = r
            d = score_delta(res)
            if d > best[0]: best = (d, th)
        b = fit(np.concatenate([fx(m, True) for m in mtr]), np.concatenate([m['y'] for m in mtr]))
        pte = {m['movie']: b.predict(fx(m, False)) for m in mte}
        y = np.concatenate([m['y'] for m in mte]); p = np.concatenate([pte[m['movie']] for m in mte])
        yc = np.concatenate([m['y'] for m in mte if m['set'] in CL]); pc = np.concatenate([pte[m['movie']] for m in mte if m['set'] in CL])
        print('\n== %s -> %s  %-9s AUC %.3f (clean %.3f) | train-OOF best th %s (%+.5f) | OOF-train topK: %s' % (tr, te, vn, auc(p, y), auc(pc, yc), best[1], best[0], topk(mtr, oof)), flush=True)
        print('     test topK: %s' % topk(mte, pte), flush=True)
        for th in THS:
            res = {m['movie']: greedy(m, pte[m['movie']], th) for m in mte}
            n = sum(r[2] for r in res.values()); ny = sum(r[3] for r in res.values()); nf = sum(r[4] for r in res.values())
            cl = {k: v for k, v in res.items() if any(mm['movie'] == k and mm['set'] in CL for mm in mte)}
            print('   th %.2f%s applied %4d correct %3d (flip %3d) dTP %+4d dFP %+4d | score %+.5f | clean %+.5f' % (
                th, '*' if th == best[1] else ' ', n, ny, nf, sum(r[0] for r in res.values()), sum(r[1] for r in res.values()), score_delta(res), score_delta(cl)), flush=True)
            if th == best[1]: DEC.setdefault(vn, {}).update({k: v[5] for k, v in res.items()})
for vn, d in DEC.items(): pickle.dump(d, open('%s/out/dec_%s_%s_%s.pkl' % (LN, TAG, vn, '-'.join(DIRS)), 'wb'))
print('decisions written', list(DEC))
