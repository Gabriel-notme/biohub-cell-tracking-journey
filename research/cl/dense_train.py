"""Dense relink: extract candidates with GT gain labels on P7e graphs, train (training movies only), evaluate.
usage: dense_train.py extract <tag> <list> <graph_dir> <dense_dir> <fg_dir>
       dense_train.py train
       dense_train.py eval <cfg> <list> <graph_dir> <dense_dir> <fg_dir> <th> [el2]"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
OUT = Path('/workspace/cl/dcands')


def ex_job(a):
    tag, name, g, dd, fg = a
    import evalx, dense_relink, edge_link
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(g) / (name + '.json'))
    dp = dense_relink.load_dense(Path(dd) / (name + '.npz'), nodes)
    f = edge_link.load_full(Path(fg) / (name + '.geff')); fe = {(int(x), int(y)): float(p) for (x, y), p in zip(f[3].tolist(), f[4].tolist())}
    rows = dense_relink.features(nodes, edges, dp, fe)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); gs = defaultdict(set); gp = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].add(int(y)); gp[int(y)] = int(x)
    tp = lambda a_, b_: int(p2g.get(a_) is not None and p2g.get(b_) is not None and p2g[b_] in gs.get(p2g[a_], ()))
    valid = lambda a_, b_: int((p2g.get(a_) is not None and len(gs.get(p2g[a_], ())) > 0) or (p2g.get(b_) is not None and p2g[b_] in gp))
    for r in rows:
        rem = [(r['s'], r['cur_d'])] if r['cur_d'] is not None else []
        if r['cur_s'] is not None: rem.append((r['cur_s'], r['d']))
        add = [(r['s'], r['d'])]
        g_ = (sum(tp(*e) for e in add) - sum(tp(*e) for e in rem)) - 0.94 * (sum(valid(*e) - tp(*e) for e in add) - sum(valid(*e) - tp(*e) for e in rem))
        r['gain'] = g_; r['lab'] = 'P' if g_ > .01 else ('N' if g_ < -.01 else 'U'); r['movie'] = name; r['set'] = tag
    OUT.mkdir(exist_ok=True)
    (OUT / ('%s__%s.json' % (tag, name))).write_text(json.dumps(rows))
    return len(rows), sum(r['lab'] == 'P' for r in rows), sum(r['lab'] == 'N' for r in rows), len(dp)


def ev_job(a):
    cfg, name, g, dd, fg, th, el2 = a
    import evalx, dense_relink, edge_link, prune
    nodes, edges = evalx.load_graph_json(Path(g) / (name + '.json'))
    n2, e2, st = dense_relink.apply(nodes, edges, Path(dd) / (name + '.npz'), Path(fg) / (name + '.geff'), '/workspace/cl/dense_lgb.json', th=th)
    if el2:
        n2, e2, s2 = edge_link.apply(n2, e2, Path(fg) / (name + '.geff'), '/workspace/cl/edge_lgb.json', th=0.4); st.update(s2)
        n2, e2, s3 = prune.prune_fragments(n2, e2, 2)
    pairs = [(int(e['source_id']), int(e['target_id'])) for e in e2]
    assert len(pairs) == len(set(pairs)) and max(Counter(t for _, t in pairs).values()) <= 1 and max(Counter(s for s, _ in pairs).values()) <= 2
    r = evalx.score_movie(name, n2, e2); r['movie'] = name; r['stats'] = {k: v for k, v in st.items() if isinstance(v, (int, float))}
    return r


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'extract':
        tag, lst, g, dd, fg = sys.argv[2:7]
        names = [l.strip() for l in open(lst) if l.strip()]
        with Pool(32) as pool: res = pool.map(ex_job, [(tag, n, g, dd, fg) for n in names])
        print(tag, 'rows %d P %d N %d dense pairs %d' % tuple(sum(x[i] for x in res) for i in range(4)))
    elif mode == 'train':
        import lightgbm as lgb, dense_relink
        rows = []
        for f in glob.glob(str(OUT / 'dtrain__*.json')): rows += json.load(open(f))
        feats = dense_relink.FEATS
        X = np.array([[(-1 if r.get(k) is None else r.get(k, -1)) for k in feats] for r in rows], np.float32)
        lab = np.array([r['lab'] for r in rows]); mv = np.array([r['movie'] for r in rows]); gain = np.array([r['gain'] for r in rows]); y = (lab == 'P').astype(int)
        tr = lab != 'U'
        um = np.unique(mv); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
        params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=0, num_threads=16)
        oof = np.full(len(rows), np.nan)
        for k in range(5):
            trk = tr & np.array([fold[m] != k for m in mv]); vak = np.array([fold[m] == k for m in mv])
            oof[vak] = lgb.train(params, lgb.Dataset(X[trk], y[trk]), 400).predict(X[vak])
        b = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 400)
        json.dump(b.dump_model(), open('/workspace/cl/dense_lgb.json', 'w'))
        def auc(pp, yy):
            o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
            return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)
        print('train rows', len(rows), 'P', y.sum(), 'N', (lab == 'N').sum(), 'OOF AUC %.3f' % auc(oof[tr], y[tr]), 'dp-only AUC %.3f' % auc(X[tr, feats.index('dp')], y[tr]))
        for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            m = oof >= th; print('  th %.1f OOF n %d P %d N %d gain %+.1f' % (th, m.sum(), y[m].sum(), (lab[m] == 'N').sum(), gain[m].sum()))
        imp = sorted(zip(b.feature_importance('gain'), feats), reverse=True); print([(f, int(g)) for g, f in imp[:8]])
    else:
        cfg, lst, g, dd, fg, th = sys.argv[2:8]; el2 = len(sys.argv) > 8
        names = [l.strip() for l in open(lst) if l.strip()]
        with Pool(32) as pool: rows = pool.map(ev_job, [(cfg, n, g, dd, fg, float(th), el2) for n in names])
        tot = Counter()
        for r in rows: tot.update(r['stats'])
        for s, ns in [('hold36', [l.strip() for l in open('/workspace/hold36.txt') if l.strip()]), ('prev4', [l.strip() for l in open('/workspace/preview4.txt') if l.strip()])]:
            rr = [r for r in rows if r['movie'] in ns]
            if rr: json.dump(rr, open('/workspace/cl/rev/%s_%s.json' % (cfg, s), 'w'))
        print(cfg, dict(tot))
