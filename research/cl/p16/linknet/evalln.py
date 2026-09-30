"""Held-out-embryo evaluation of linknet pair scores on the rl_cands evaluable rows.
usage: evalln.py <tag_trained_on_44b6(scores 6bba)> <tag_trained_on_6bba(scores 44b6)>"""
import os, sys, glob, json, pickle
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from tracking_cellmot.metrics import summarise
TA, TB = sys.argv[1], sys.argv[2]
SRC = {'6bba': TA, '44b6': TB}
CL = ('hold36', 'prev4')
base = {}; FLIP = {}
for f in glob.glob('/workspace/cl/nm/rl_oracle/*.json'):
    d = json.load(open(f)); base[d['movie']] = d['scores']['base']
    FLIP[d['movie']] = {(x['p1'], x['p2']): x['flip'] for x in d['fixes']}
GP = pickle.load(open('/workspace/cl/p16/linknet/geo_pred.pkl', 'rb')) if os.path.exists('/workspace/cl/p16/linknet/geo_pred.pkl') else {}


def logsig(a): return -np.logaddexp(0, -a)


def log1msig(a): return -np.logaddexp(0, a)


def auc(p, y):
    o = np.argsort(p, kind='mergesort'); rk = np.empty(len(p)); rk[o] = np.arange(len(p)); n1 = y.sum(); n0 = len(y) - n1
    return (rk[y == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


def score_delta(res):
    ms = list(res); a = summarise([base[m] for m in ms]); out = []
    for m in ms:
        b = dict(base[m]); dtp, dfp = res[m]
        b['edge_tp'] += int(dtp); b['edge_fp'] += int(dfp); b['edge_fn'] -= int(dtp)
        W = b['edge_tp'] + b['edge_fp'] + b['edge_fn']; b['adj_edge_jaccard'] = max(0., b['edge_tp'] / W * (1 - 0.1 * b['total_node_ratio'])); out.append(b)
    return summarise(out)['score'] - a['score']


def load_emb(emb, src):
    D = []
    for f in sorted(glob.glob('/workspace/cl/p16/linknet/tab/*%s_*.pkl' % emb)):
        s, m = os.path.basename(f)[:-4].split('__'); tb = pickle.load(open(f, 'rb'))
        if len(tb['rows']) == 0: continue
        z = np.load('/workspace/cl/p16/linknet/out/%s/scores/%s__%s.npz' % (src, s, m)); P = z['P']; lg = z['lg']
        key = P[:, 0] * 10 ** 7 + P[:, 1]; o = np.argsort(key); key = key[o]; lg = lg[o]

        def look(a, b):
            k = np.maximum(a, 0) * 10 ** 7 + np.maximum(b, 0); j = np.clip(np.searchsorted(key, k), 0, len(key) - 1)
            ok = (a >= 0) & (b >= 0) & (key[j] == k)
            return np.where(ok, lg[j], np.nan)
        rw = tb['rows']; S_, D_, C_, Q_ = rw[:, 0], rw[:, 1], rw[:, 2], rw[:, 3]
        asd = look(S_, D_); asc = look(S_, C_); aqd = look(Q_, D_)
        A1 = logsig(asd) + np.where(np.isnan(asc), 0, log1msig(np.nan_to_num(asc))) + np.where(np.isnan(aqd), 0, log1msig(np.nan_to_num(aqd)))
        A2 = asd - np.fmax(np.nan_to_num(asc, nan=-30), np.nan_to_num(aqd, nan=-30))
        b1 = pickle.load(open('/workspace/cl/nm/rl_b1f/%s__%s.pkl' % (s, m), 'rb'))['X'][:, 0]
        g = GP.get(m, {}); ids = tb['ids']
        fl = np.array([FLIP[m].get((int(ids[a]), int(ids[b])), 0) for a, b in zip(S_, D_)], int) * tb['y'].astype(int)
        D.append(dict(set=s, movie=m, ids=ids, S=S_, D=D_, C=C_, Q=Q_, y=tb['y'].astype(int), flip=fl, dtp=tb['dtp'].astype(int), dfp=tb['dfp'].astype(int), typ=tb['typ'],
                      scv={'ln_sd': asd, 'ln_A1': A1, 'ln_A2': A2, 'b1_sd': b1, **{k: v for k, v in g.items()}}))
    return D


def greedy_topk(D, k, Ks=(10, 30, 50, 100, 200)):
    allv = np.concatenate([d['scv'][k] for d in D]); mix = np.concatenate([np.full(len(d['y']), i) for i, d in enumerate(D)])
    rix = np.concatenate([np.arange(len(d['y'])) for d in D])
    o = np.argsort(-allv, kind='mergesort'); used = [set() for _ in D]; res = {d['movie']: [0, 0] for d in D}; n = 0; ny = 0; nf = 0; out = []
    for i in o:
        d = D[mix[i]]; j = rix[i]; ks = [x for x in (d['S'][j], d['D'][j], d['C'][j], d['Q'][j]) if x >= 0]
        if any(x in used[mix[i]] for x in ks): continue
        used[mix[i]].update(ks); n += 1; ny += d['y'][j]; nf += d['flip'][j]; res[d['movie']][0] += d['dtp'][j]; res[d['movie']][1] += d['dfp'][j]
        if n in Ks:
            dt = sum(v[0] for v in res.values()); df = sum(v[1] for v in res.values())
            out.append('K%d prec %.2f (real %d flip %d) dTP %+d dFP %+d d %+.5f' % (n, ny / n, ny - nf, nf, dt, df, score_delta(res)))
        if n >= max(Ks): break
    return out


def thresh_apply(D, k, th, prob=True):
    res = {d['movie']: [0, 0] for d in D}; n = ny = 0
    for d in D:
        p = np.exp(d['scv'][k]) if prob else d['scv'][k]; o = np.argsort(-p); used = set()
        for j in o:
            if p[j] < th: break
            ks = [x for x in (d['S'][j], d['D'][j], d['C'][j], d['Q'][j]) if x >= 0]
            if any(x in used for x in ks): continue
            used.update(ks); n += 1; ny += d['y'][j]; res[d['movie']][0] += d['dtp'][j]; res[d['movie']][1] += d['dfp'][j]
    return res, n, ny


if __name__ == '__main__':
    R = {}
    for emb in ['44b6', '6bba']:
        D = load_emb(emb, SRC[emb]); R[emb] = D
        keys = list(D[0]['scv'])
        print('\n==== held-out %s (model trained on the other embryo) movies %d rows %d pos %d' % (emb, len(D), sum(len(d['y']) for d in D), sum(d['y'].sum() for d in D)))
        for grp, fn in [('all', lambda d: True), ('clean40', lambda d: d['set'] in CL), ('b1-insample', lambda d: d['set'] not in CL)]:
            sel = [d for d in D if fn(d)]; y = np.concatenate([d['y'] for d in sel]); typ = np.concatenate([d['typ'] for d in sel])
            line = '  %-12s pos %4d |' % (grp, y.sum())
            for k in keys:
                p = np.concatenate([d['scv'][k] for d in sel]); line += ' %s %.3f' % (k, auc(p, y))
            print(line)
            if grp == 'all':
                for t in range(4):
                    kk = typ == t; line = '     typ %d rows %6d pos %4d |' % (t, kk.sum(), y[kk].sum())
                    for k in keys:
                        p = np.concatenate([d['scv'][k] for d in sel]); line += ' %s %.3f' % (k, auc(p[kk], y[kk]))
                    print(line)
        for k in keys:
            print('   %-7s' % k, ' | '.join(greedy_topk(D, k)))
        for th in (0.5, 0.7, 0.9):
            res, n, ny = thresh_apply(D, 'ln_A1', th)
            cl = {d['movie']: res[d['movie']] for d in D if d['set'] in CL}
            print('   A1>=%.1f applied %d correct %d dTP %+d dFP %+d | score %+.5f | clean %+.5f' % (th, n, ny, sum(v[0] for v in res.values()), sum(v[1] for v in res.values()), score_delta(res), score_delta(cl)))
    pickle.dump(R, open('/workspace/cl/p16/linknet/out/eval_%s_%s.pkl' % (TA, TB), 'wb'))
