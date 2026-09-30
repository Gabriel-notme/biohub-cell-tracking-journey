"""Is the (less negative) LOEO result on the 20 clean 6bba movies special, or within the spread of random 20-movie subsets of 6bba?
Also: relative annotation rate of the removed tracks (removed TP/node divided by movie mean TP/node)."""
import pickle, numpy as np
R = pickle.load(open('var_preds.pkl', 'rb'))
B6 = [r for r in R if r['movie'].startswith('6bba')]


def mg(r, a, key='x_FULL'):
    p = r[key]; sel = (p / r['len']) < a * p.sum() / r['len'].sum()
    m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']
    n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum()
    return np.array([r['TP'] * m, w, (r['TP'] - k) * (m + 0.1 * n / r['ntot']), w - f, n, k, r['tp'].sum(), r['len'].sum()])


def g(P): return P[:, 2].sum() / P[:, 3].sum() - P[:, 0].sum() / P[:, 1].sum()


rng = np.random.default_rng(0)
for a in [0.03, 0.05, 0.1]:
    P = np.array([mg(r, a) for r in B6]); clean = np.array([r['set'] in ('p8_hold36', 'p8_prev4') for r in B6])
    sims = np.array([g(P[rng.choice(len(B6), 20, replace=False)]) for _ in range(5000)])
    gc = g(P[clean]); pct = (sims <= gc).mean()
    rel_c = (P[clean, 5].sum() / P[clean, 4].sum()) / (P[clean, 6].sum() / P[clean, 7].sum())
    rel_o = (P[~clean, 5].sum() / P[~clean, 4].sum()) / (P[~clean, 6].sum() / P[~clean, 7].sum())
    print('a%.2f all128 %+.5f | clean20 %+.5f is at percentile %.2f of random 20-subsets (median %+.5f, 5%% %+.5f, 95%% %+.5f) | removed-track TP rate / mean: clean %.2f other %.2f (break-even ~%.2f)'
          % (a, g(P), gc, pct, np.median(sims), np.percentile(sims, 5), np.percentile(sims, 95), rel_c, rel_o, 0.1))
