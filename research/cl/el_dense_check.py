"""Does the dense transformer probability improve free-end linking? OOF comparison on the dense training movies,
using P7e-order graphs before linking is not available, so use P7e final graphs' remaining free ends."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from multiprocessing import Pool
import numpy as np


def job(name):
    import evalx, edge_cands, dense_relink
    # P3-stage graph = B5 + dc/dsr/dfork (no links, no prune) is what edge_link sees in P7e; rebuild it from the dtrain P7e run's
    # inputs is costly, so approximate with the B5 lineage graph of the dense run (free ends before any P-stage edits)
    nodes, edges = evalx.load_graph_json(Path('/workspace/runs/b5d_train/working/lineage_graphs') / (name + '.json'))
    fp = Path('/workspace/runs/fg2_train') / (name + '.geff')
    rows = edge_cands.features(nodes, edges, fp)
    edge_cands.label(name, nodes, edges, rows)
    dp = dense_relink.load_dense(Path('/workspace/runs/dense_train') / (name + '.npz'), nodes)
    into = {}
    for (a, b), p in dp.items(): into[b] = max(into.get(b, 0.), p)
    for r in rows:
        r['dp'] = dp.get((r['s'], r['d']), 0.) if r['gap'] == 1 else -1.; r['dp_best_d'] = into.get(r['d'], 0.); r['movie'] = name
    return [r for r in rows if r['lab'] != 'U']


if __name__ == '__main__':
    import lightgbm as lgb, edge_link
    names = [l.strip() for l in open('/workspace/cl/dense_train.txt') if l.strip()]
    with Pool(32) as pool: rows = [r for rs in pool.map(job, names) for r in rs]
    y = np.array([r['lab'] == 'P' for r in rows], int); mv = np.array([r['movie'] for r in rows])
    um = np.unique(mv); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=16)
    def auc(pp, yy):
        o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
        return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)
    for nm, feats in [('base', edge_link.FEATS), ('+dense', edge_link.FEATS + ['dp', 'dp_best_d'])]:
        X = np.array([[(-1 if r.get(k) is None else r.get(k, -1)) for k in feats] for r in rows], np.float32)
        oof = np.zeros(len(rows))
        for k in range(5):
            tr = np.array([fold[m] != k for m in mv]); va = ~tr
            oof[va] = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 400).predict(X[va])
        best = max((sum(y[oof >= t] - 0.94 * (1 - y[oof >= t])), t) for t in np.arange(0.2, 0.91, 0.05))
        print(nm, 'n', len(rows), 'P', y.sum(), 'OOF AUC %.4f' % auc(oof, y), 'best OOF gain %.1f at th %.2f' % best)
