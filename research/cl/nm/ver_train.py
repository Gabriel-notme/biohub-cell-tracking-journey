"""ver_train: leave-one-embryo-out verifier for candidate additions (ver_cands). Trains on evaluable candidates (TP=1, FP=0) of one
embryo, scores every candidate of the other embryo. Threshold chosen on the TRAINING embryo only (movie-grouped 5-fold OOF scores),
maximising  sum(TP) - 0.94*sum(FP) - CN*nodes  over accepted candidates (walk prefix semantics approximated per candidate).
usage: ver_train.py <tag> [extra feature json: {movie: {key: [f...]}}] [extra names comma]"""
import os, sys, json, glob
os.environ['OMP_NUM_THREADS'] = '1'
from collections import defaultdict
import numpy as np
import lightgbm as lgb
tag = sys.argv[1]
EXTRA = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[2] != '-' else None
ENAMES = sys.argv[3].split(',') if len(sys.argv) > 3 else []
CN = float(os.environ.get('VER_CN', '0.003'))
FAMS = os.environ.get('VER_FAMS', 'walk,join,near').split(',')

rows = []
for f in sorted(glob.glob('/workspace/cl/nm/ver_cands/*/*.json')):
    d = json.load(open(f)); m = d['movie']
    for w in d['walks']:
        n, fwd = w['seed'], w['fwd']
        T0, T1 = w['T0'], w['T1']; tb = min(w['t0'] - T0, T1 - w['t0'])
        base = [fwd, min(w['L'], 50), w['vlen'], tb]
        for k, s in enumerate(w['steps']):
            key = '%d_%d_s%d' % (n, fwd, k)
            ft = base + [0, k, s['p'], s['p2'], s['nalt'], s['ilpn'], s['ilpe'], s['dup'], s['dist'], s['vdev'], s['dens'], len(w['steps']),
                         int(w['join'] is not None)]
            rows.append((m, key, 'walk', s['lab'], ft))
        if w['join']:
            j = w['join']; key = '%d_%d_j' % (n, fwd)
            ft = base + [1, len(w['steps']), j['p'], 0, 0, 1, 1, 99, j['dist'], j['dist'], 0, len(w['steps']), 1]
            rows.append((m, key, 'join', j['lab'], ft))
        if w['near']:
            s = w['near']; key = '%d_%d_n' % (n, fwd)
            ft = base + [2, 0, s['p'], 0, s['pedge'], s['ilpn'], 0, s['dup'], s['raw'], s['dist'], s['dens'], 0, 0]
            rows.append((m, key, 'near', s['lab'], ft))
FN = ['fwd', 'L', 'vlen', 'tbound', 'fam', 'k', 'p', 'p2', 'nalt', 'ilpn', 'ilpe', 'dup', 'dist', 'vdev', 'dens', 'nsteps', 'hasjoin'] + ENAMES
rows = [r for r in rows if r[2] in FAMS]
if EXTRA is not None:
    rows = [(m, k, fa, l, ft + EXTRA.get(m, {}).get(k, [np.nan] * len(ENAMES))) for m, k, fa, l, ft in rows]
X = np.array([r[4] for r in rows], float); y = np.array([1 if r[3] == 'TP' else 0 for r in rows]); ev = np.array([r[3] != 'NE' for r in rows])
emb = np.array([r[0][:4] for r in rows]); mov = np.array([r[0] for r in rows])
print('rows', len(rows), 'evaluable', ev.sum(), 'TP', y.sum(), {e: (int((ev & (emb == e)).sum()), int((y[emb == e]).sum())) for e in ['44b6', '6bba']})
P = dict(objective='binary', learning_rate=0.03, num_leaves=7, min_data_in_leaf=15, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, num_threads=4, seed=0)
NR = int(os.environ.get('VER_NR', '250'))


def fit(idx):
    return lgb.train(P, lgb.Dataset(X[idx], y[idx]), NR)


def auc(s, l):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(l, s) if 0 < l.sum() < len(l) else float('nan')


def gain_at(s, lab_ev, lab_y, th):
    a = s >= th
    return (lab_y[a & lab_ev].sum() - 0.94 * ((1 - lab_y[a & lab_ev]).sum()) - CN * a.sum()), int(a.sum()), int(lab_y[a & lab_ev].sum()), int(((1 - lab_y)[a & lab_ev]).sum())


scores = {}; report = {}
for test in ['44b6', '6bba']:
    tr = emb != test; te = emb == test
    # inner OOF on training embryo, movie-grouped
    ms = sorted(set(mov[tr])); rng = np.random.default_rng(0); rng.shuffle(ms); fold = {m: i % 5 for i, m in enumerate(ms)}
    oof = np.full(len(rows), np.nan)
    for f in range(5):
        itr = np.where(tr & ev & np.array([fold.get(m, -1) != f for m in mov]))[0]
        ite = np.where(tr & np.array([fold.get(m, -1) == f for m in mov]))[0]
        mdl = fit(itr); oof[ite] = mdl.predict(X[ite])
    ths = np.quantile(oof[tr], np.linspace(0.5, 0.99995, 400))
    best = max(ths, key=lambda th: gain_at(oof[tr], ev[tr], y[tr], th)[0])
    g_tr = gain_at(oof[tr], ev[tr], y[tr], best)
    mdl = fit(np.where(tr & ev)[0]); s = mdl.predict(X[te])
    g_te = gain_at(s, ev[te], y[te], best)
    orc = y[te & ev].sum()
    imp = sorted(zip(mdl.feature_importance('gain'), FN), reverse=True)[:8]
    report[test] = dict(auc_oof_train=round(auc(oof[tr & ev], y[tr & ev]), 3), auc_test=round(auc(s[ev[te]], y[te][ev[te]]), 3), th=float(best),
                        train_gain=g_tr, test_gain=g_te, test_oracle_TP=int(orc),
                        auc_test_by_fam={fa: round(auc(s[ev[te] & (np.array([r[2] for r in rows])[te] == fa)], y[te][ev[te] & (np.array([r[2] for r in rows])[te] == fa)]), 3) for fa in FAMS},
                        imp=[(n, int(g)) for g, n in imp])
    print(test, json.dumps(report[test]), flush=True)
    for i, sc in zip(np.where(te)[0], s): scores.setdefault(rows[i][0], {})[rows[i][1]] = float(sc)
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print('   fixed th %.1f test gain/acc/TP/FP' % th, gain_at(s, ev[te], y[te], th))
json.dump(scores, open('/workspace/cl/nm/ver_scores_%s.json' % tag, 'w'))
json.dump(report, open('/workspace/cl/nm/ver_report_%s.json' % tag, 'w'))
