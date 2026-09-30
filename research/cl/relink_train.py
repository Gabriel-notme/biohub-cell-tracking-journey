"""Extract relink candidates on P3 graphs for all sets, label with GT, train LightGBM on t127a+t127b+audit32."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
SETS = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/p3_hold36/graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/runs/p3_prev4/graphs', '/workspace/runs/fullgraph_prev4'),
        'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/p3_audit32/graphs', '/workspace/sync3/runs/fullgraph_audit32'),
        't127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/fin_p3_t127a/graphs', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/t127b.txt', '/workspace/sync3/runs/fin_p3_t127b/graphs', '/workspace/sync3/runs/fullgraph_t127b')}
OUT = Path('/workspace/cl/rcands')


def job(a):
    s, name, gdir, fdir = a
    import evalx, relink, edge_link
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    rows = relink.features(nodes, edges, edge_link.load_full(Path(fdir) / (name + '.geff')))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(x)].append(int(y)); gpar[int(y)] = int(x)
    for r in rows:
        gs, gd = p2g.get(r['s']), p2g.get(r['d'])
        valid = (gs is not None and len(gsucc.get(gs, [])) > 0) or (gd is not None and gd in gpar)
        ok = gs is not None and gd is not None and gd in gsucc.get(gs, [])
        # is the current s->cur_d link correct?
        gc = p2g.get(r['cur_d']) if r['cur_d'] is not None else None
        r['cur_ok'] = int(gs is not None and gc is not None and gc in gsucc.get(gs, []))
        r['lab'] = 'P' if ok else ('N' if valid else 'U')
        r['set'] = s; r['movie'] = name
    OUT.mkdir(exist_ok=True)
    (OUT / ('%s__%s.json' % (s, name))).write_text(json.dumps(rows))
    return len(rows)


if __name__ == '__main__':
    if not (len(sys.argv) > 1 and sys.argv[1] == 'train'):
        jobs = [(s, n, g, f) for s, (lst, g, f) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
        with Pool(40) as pool: print('rows', sum(pool.map(job, jobs)))
    import lightgbm as lgb, relink
    FEATS = relink.FEATS
    rows = []
    for f in glob.glob(str(OUT / '*.json')): rows += json.load(open(f))
    X = np.array([[r.get(k, -1) if r.get(k) is not None else -1 for k in FEATS] for r in rows], dtype=np.float32)
    lab = np.array([r['lab'] for r in rows]); sets = np.array([r['set'] for r in rows]); movies = np.array([r['movie'] for r in rows])
    y = (lab == 'P').astype(int)
    from collections import Counter
    print(Counter(zip(sets, lab)))
    print('typ x lab', Counter((r['typ'], r['lab']) for r in rows if r['set'] in ('hold36',)))
    tr = np.isin(sets, ['t127a', 't127b', 'audit32']) & (lab != 'U')
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=8)
    um = np.unique(movies[tr]); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
    oof = np.full(len(rows), np.nan)
    for k in range(5):
        trk = tr & np.array([fold.get(m, -1) != k for m in movies]); vak = tr & np.array([fold.get(m, -1) == k for m in movies])
        oof[vak] = lgb.train(params, lgb.Dataset(X[trk], y[trk]), 400).predict(X[vak])
    bst = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 400); bst.save_model('/workspace/cl/relink_lgb.txt')
    p = bst.predict(X)
    def auc(pp, yy):
        o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
        return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)
    print('OOF AUC %.3f' % auc(oof[tr], y[tr]))
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mo = tr & (oof >= th); line = '%.2f OOF prec %.3f n %d/%d |' % (th, y[mo].mean() if mo.sum() else 0, y[mo].sum(), mo.sum())
        for s in ['hold36', 'prev4']:
            m = (sets == s) & (lab != 'U'); mm = m & (p >= th)
            line += ' %s prec %.3f (%d/%d of %d)' % (s, y[mm].mean() if mm.sum() else 0, y[mm].sum(), mm.sum(), y[m].sum())
        print(line)
    imp = sorted(zip(bst.feature_importance('gain'), FEATS), reverse=True); print([(f, int(g)) for g, f in imp[:10]])
