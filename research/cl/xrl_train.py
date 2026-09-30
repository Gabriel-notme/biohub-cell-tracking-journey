"""Leave-one-embryo-out training of the extended relinker on P13 residual candidates.
Train only on evaluable rows (new or removed edge touches an annotated track); target = pos & !old_tp (pure gain).
Held-out evaluation: greedy application per movie (same conflict rule as xrl.apply_rows); exact edge deltas from labels:
dTP = pos - old_tp ; dFP = (new_ev & !pos) - (old_ev & !old_tp). Converted to score deltas with the per-movie P13 rows."""
import os, sys, json, glob, pickle
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src')
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from xrl import FEATS
SRC = os.environ.get('XRL_SRC', 'p13')
rows = []
for f in sorted(glob.glob('/workspace/cl/xrl_%s/*.pkl' % SRC)):
    s, m = os.path.basename(f)[:-4].split('__')
    for r in pickle.load(open(f, 'rb')): r['set'] = s; r['movie'] = m; r['emb'] = m[:4]; rows.append(r)
base = {}; CLEANM = set()
for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']:
    for r in json.load(open('/workspace/cl/rev/p13r_%s.json' % s)):
        base[r['movie']] = r
        if s in ('hold36', 'prev4'): CLEANM.add(r['movie'])
from tracking_cellmot.metrics import summarise


def score_rows(movies, d):
    out = []
    for m in movies:
        r = dict(base[m]); dtp, dfp = d.get(m, (0, 0))
        r['edge_tp'] += dtp; r['edge_fn'] -= dtp; r['edge_fp'] += dfp
        den = r['edge_tp'] + r['edge_fp'] + r['edge_fn']
        r['edge_jaccard'] = r['edge_tp'] / den if den else 0.
        fac = 1 - 0.1 * (r['num_pred_nodes'] - r['n_total']) / r['n_total']
        r['adj_edge_jaccard'] = max(0., r['edge_jaccard'] * fac)
        out.append(r)
    return summarise(out)


# sanity: recomputation reproduces base
ms = sorted(base); a = summarise([base[m] for m in ms]); b = score_rows(ms, {})
print('sanity base %.6f recomputed %.6f' % (a['score'], b['score']))
X = np.array([[r.get(k, -1) for k in FEATS] for r in rows], np.float32)
ev = np.array([r['new_ev'] or r['old_ev'] for r in rows], bool)
y = np.array([r['pos'] and not r['old_tp'] for r in rows], int)
emb = np.array([r['emb'] for r in rows]); setv = np.array([r['set'] for r in rows])
print('rows %d evaluable %d pure-gain pos %d | by type: ' % (len(rows), ev.sum(), y.sum()),
      {t: (int(((X[:, 0] == t) & ev).sum()), int(((X[:, 0] == t) & (y == 1)).sum())) for t in [0, 1]})
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, seed=0, num_threads=16)
NR = int(os.environ.get('XRL_NR', '300'))
res = {}
for tr_e, te_e in [('44b6', '6bba'), ('6bba', '44b6')]:
    tr = ev & (emb == tr_e)
    bst = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), NR)
    te = np.where(emb == te_e)[0]
    p = bst.predict(X[te])
    bym = defaultdict(list)
    for i, q in zip(te, p): bym[rows[i]['movie']].append((q, i))
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        d = {}; nap = 0; G = B = 0
        for m, lst in bym.items():
            lst.sort(reverse=True); touched = set(); dtp = dfp = 0
            for q, i in lst:
                if q < th: break
                r = rows[i]
                if r['s'] in touched or r['d'] in touched or r['o'] in touched: continue
                touched.update((r['s'], r['d'], r['o'])); nap += 1
                dtp += r['pos'] - r['old_tp']; dfp += int(r['new_ev'] and not r['pos']) - int(r['old_ev'] and not r['old_tp'])
            d[m] = (dtp, dfp); G += dtp; B += dfp
        mm = sorted(m for m in base if m.startswith(te_e))
        sa, sb = summarise([base[m] for m in mm]), score_rows(mm, d)
        line = 'train %s -> test %s th %.1f applied %5d dTP %+4d dFP %+4d | score d %+.5f' % (tr_e, te_e, th, nap, G, B, sb['score'] - sa['score'])
        cm = [m for m in mm if m in CLEANM]
        if cm:
            ca, cb = summarise([base[m] for m in cm]), score_rows(cm, d)
            line += ' | clean(%d) d %+.5f' % (len(cm), cb['score'] - ca['score'])
        print(line, flush=True)
    imp = sorted(zip(bst.feature_importance('gain'), FEATS), reverse=True)[:8]
    print('   top feats', [(f, int(g)) for g, f in imp])
