"""Retrain edge_link / relink on the distribution they meet in P7e: P3 graphs BEFORE isolated-node pruning
(= P3 graph + B5-base nodes that the prune step removed). Train on t127a+t127b+audit32 only."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
SETS = {'t127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/fin_p3_t127a/graphs', '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/t127b.txt', '/workspace/sync3/runs/fin_p3_t127b/graphs', '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs', '/workspace/sync3/runs/fullgraph_t127b'),
        'audit32': ('/workspace/audit32.txt', '/workspace/cl/ps_p3_audit32/graphs', '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', '/workspace/sync3/runs/fullgraph_audit32'),
        'hold36': ('/workspace/hold36.txt', '/workspace/cl/ps_p3_hold36/graphs', '/workspace/runs/b5f_hold36/working/lineage_graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/cl/ps_p3_prev4/graphs', '/workspace/runs/b5f_prev4/working/lineage_graphs', '/workspace/runs/fullgraph_prev4')}
OUT = Path('/workspace/cl/v2cands'); OUT.mkdir(exist_ok=True)


def job(a):
    s, name, pg, bg, fg = a
    import evalx, edge_cands, relink, edge_link
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(pg) / (name + '.json'))
    bn, _ = evalx.load_graph_json(Path(bg) / (name + '.json'))
    added = 0
    for k, v in bn.items():
        if k not in nodes: nodes[k] = v; added += 1
    fp = Path(fg) / (name + '.geff')
    er = edge_cands.features(nodes, edges, fp)
    edge_cands.label(name, nodes, edges, er)
    full = edge_link.load_full(fp)
    rr = relink.features(nodes, edges, full)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(x)].append(int(y)); gpar[int(y)] = int(x)
    for r in rr:
        gs, gd = p2g.get(r['s']), p2g.get(r['d'])
        valid = (gs is not None and len(gsucc.get(gs, [])) > 0) or (gd is not None and gd in gpar)
        ok = gs is not None and gd is not None and gd in gsucc.get(gs, [])
        r['lab'] = 'P' if ok else ('N' if valid else 'U')
    for r in er + rr: r['set'] = s; r['movie'] = name
    (OUT / ('%s__%s.json' % (s, name))).write_text(json.dumps({'el': er, 'rl': rr}))
    return added


if __name__ == '__main__':
    jobs = [(s, n, pg, bg, fg) for s, (lst, pg, bg, fg) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
    with Pool(40) as pool: print('isolated nodes restored', sum(pool.map(job, jobs)))
    import lightgbm as lgb, edge_link, relink
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=8)
    D = {'el': [], 'rl': []}
    for f in glob.glob(str(OUT / '*.json')):
        d = json.load(open(f))
        for k in D: D[k] += d[k]
    for k, feats, name in [('el', edge_link.FEATS, 'edge_lgb_v2'), ('rl', relink.FEATS, 'relink_lgb_v2')]:
        rows = D[k]
        X = np.array([[(-1 if r.get(c) is None else r.get(c, -1)) for c in feats] for r in rows], np.float32)
        lab = np.array([r['lab'] for r in rows]); sets = np.array([r['set'] for r in rows]); mv = np.array([r['movie'] for r in rows]); y = (lab == 'P').astype(int)
        tr = np.isin(sets, ['t127a', 't127b', 'audit32']) & (lab != 'U')
        um = np.unique(mv[tr]); rng = np.random.default_rng(0); rng.shuffle(um); fold = {m: i % 5 for i, m in enumerate(um)}
        oof = np.full(len(rows), np.nan)
        for f_ in range(5):
            trk = tr & np.array([fold.get(m, -1) != f_ for m in mv]); vak = tr & np.array([fold.get(m, -1) == f_ for m in mv])
            oof[vak] = lgb.train(params, lgb.Dataset(X[trk], y[trk]), 400).predict(X[vak])
        b = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 400)
        json.dump(b.dump_model(), open('/workspace/cl/%s.json' % name, 'w'))
        # OOF-optimal threshold on training only: maximise sum(y - 0.94*(1-y)) over accepted labelled rows
        best = max([(sum(y[tr & (oof >= t)] - 0.94 * (1 - y[tr & (oof >= t)])), t) for t in np.arange(0.2, 0.91, 0.05)])
        p = b.predict(X)
        def auc(pp, yy):
            o = np.argsort(pp); rr_ = np.empty(len(pp)); rr_[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
            return (rr_[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)
        print(name, 'train P/N %d/%d' % (y[tr].sum(), (tr & (y == 0)).sum()), 'OOF AUC %.3f' % auc(oof[tr], y[tr]),
              'hold36 AUC %.3f prev4 AUC %.3f' % tuple(auc(p[(sets == s) & (lab != 'U')], y[(sets == s) & (lab != 'U')]) for s in ['hold36', 'prev4']),
              'OOF-optimal th %.2f (gain %.1f)' % (best[1], best[0]))
