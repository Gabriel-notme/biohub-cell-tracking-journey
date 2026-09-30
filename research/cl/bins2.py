"""Annotated-edge rate (relative to group mean) by feature bin, split embryo x (clean hold36/prev4 vs train-set t127/audit32 graphs)."""
import pickle, numpy as np
F = ['len', 't0', 't1', 'z', 'y', 'x', 'sz', 'sy', 'sx', 'v', 'vmax', 'ep_mean', 'ep_min', 'zr', 'rr', 'lrank', 'dup5', 'dup7', 'nn_med', 'dens10', 'npred', 'ntrk', 'frac_iso']
R = pickle.load(open('var_preds.pkl', 'rb'))
L = lambda t: pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb'))
X = {}
for t in ['p8_t127a', 'p8_t127b', 'p8_audit32', 'p8_hold36', 'p8_prev4']:
    for r in L(t): X[r['movie']] = r['X']
G = {}
for r in R: G.setdefault((r['movie'][:4], 'clean' if r['set'] in ('p8_hold36', 'p8_prev4') else 'trainset'), []).append(r)


def table(fname, bins):
    print('== rate by', fname)
    for k in sorted(G):
        rs = G[k]
        v = np.concatenate([X[r['movie']][:, F.index(fname)] for r in rs]); ln = np.concatenate([r['len'] for r in rs]); tp = np.concatenate([r['tp'] for r in rs])
        mean = tp.sum() / ln.sum(); b = np.digitize(v, bins)
        print(' %s/%-8s n=%3d mean %.4f |' % (k[0], k[1], len(rs), mean), ' '.join('[%s] %.2f(%.3f)' % (('<%g' % bins[0]) if i == 0 else ('>=%g' % bins[i - 1]), tp[b == i].sum() / max(1, ln[b == i].sum()) / mean, ln[b == i].sum() / ln.sum()) for i in range(len(bins) + 1)))


table('len', [5, 10, 20, 40, 80])
table('dens10', [1, 2, 3, 4])
table('dup5', [0.01, 0.1, 0.3])
table('rr', [20, 40, 60, 80, 100, 120])
table('ep_mean', [0.7, 0.8, 0.9, 0.95])
table('nn_med', [5, 7, 9, 12, 20])
# which tracks does the 44b6 model put at the bottom (a=0.05) in each 6bba group?
print('== 6bba tracks removed by 44b6-model relative rule a=0.05: feature medians of removed vs all')
for grp in ['clean', 'trainset']:
    rs = G[('6bba', grp)]
    sel = np.concatenate([(r['x_FULL'] / r['len']) < 0.05 * r['x_FULL'].sum() / r['len'].sum() for r in rs])
    XX = np.concatenate([X[r['movie']] for r in rs]); tp = np.concatenate([r['tp'] for r in rs])
    print(' %-8s removed %d tracks (tp %d) |' % (grp, sel.sum(), tp[sel].sum()), ' '.join('%s %.3g/%.3g' % (f, np.median(XX[sel, F.index(f)]), np.median(XX[:, F.index(f)])) for f in ['len', 't0', 'z', 'rr', 'ep_mean', 'dup5', 'nn_med', 'dens10', 'npred']))
