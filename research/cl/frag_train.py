"""Extract fragment gap-filling candidates on P5-style graphs, label with GT, train cross-fitted models."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
SETS = {'t127a': ('/workspace/t127a.txt', '/workspace/cl/p5tr/t127a', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/t127b.txt', '/workspace/cl/p5tr/t127b', '/workspace/sync3/runs/fullgraph_t127b'),
        'audit32': ('/workspace/audit32.txt', '/workspace/cl/p5tr/audit32', '/workspace/sync3/runs/fullgraph_audit32'),
        'hold36': ('/workspace/hold36.txt', '/workspace/cl/ps_p5_hold36/graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/cl/ps_p5_prev4/graphs', '/workspace/runs/fullgraph_prev4')}
OUT = Path('/workspace/cl/fcands')


def job(a):
    s, name, g, f = a
    import evalx, edge_link, frag_link
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(g) / (name + '.json'))
    full = edge_link.load_full(Path(f) / (name + '.geff'))
    fids, fT, fV, fE, fprob = full; idx = {int(i): j for j, i in enumerate(fids.tolist())}
    rows = frag_link.features(nodes, edges, full)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    matched_g = set(p2g.values())
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    ug = defaultdict(list)
    for i, t, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list()):
        if int(i) not in matched_g: ug[int(t)].append((int(i), np.array([z, y, x]) * S))
    utree = {t: ([i for i, _ in v], cKDTree(np.array([p for _, p in v]))) for t, v in ug.items()}
    ea = gt.edge_attrs(); gsucc = defaultdict(set); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(x)].add(int(y)); gpar[int(y)] = int(x)
    for r in rows:
        hit = []
        for x in r['chain']:
            t = int(fT[idx[x]]); p = fV[idx[x]] * S; h = None
            if t in utree:
                d, j = utree[t][1].query(p)
                if d <= 7: h = utree[t][0][int(j)]
            hit.append(h)
        tp = sum(1 for a_, b_ in zip(hit[:-1], hit[1:]) if a_ is not None and b_ is not None and b_ in gsucc.get(a_, ()))
        fp = 0
        if r['e'] is not None:
            ge = p2g.get(r['e'])
            if ge is not None and hit[0] is not None and hit[0] in gsucc.get(ge, ()): tp += 1
            elif ge is not None and gsucc.get(ge): fp += 1
        if r['s'] is not None:
            gs_ = p2g.get(r['s'])
            if gs_ is not None and hit[-1] is not None and gs_ in gsucc.get(hit[-1], ()): tp += 1
            elif gs_ is not None and gs_ in gpar: fp += 1
        r['hits'] = sum(h is not None for h in hit); r['tp'] = tp; r['fp'] = fp
        r['lab'] = 'P' if tp >= 1 else ('N' if fp >= 1 else 'U')
        r['set'] = s; r['movie'] = name; r['n_total'] = n_total
        del r['chain']
    OUT.mkdir(exist_ok=True)
    (OUT / ('%s__%s.json' % (s, name))).write_text(json.dumps(rows))
    return s, Counter(r['lab'] for r in rows), sum(r['tp'] for r in rows), sum(r['L'] for r in rows)


if __name__ == '__main__':
    if not (len(sys.argv) > 1 and sys.argv[1] == 'train'):
        jobs = [(s, n, g, f) for s, (lst, g, f) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
        tot = Counter()
        with Pool(40) as pool:
            for s, c, tp, L in pool.imap_unordered(job, jobs):
                for k, v in c.items(): tot[(s, k)] += v
                tot[(s, 'tp')] += tp; tot[(s, 'nodes')] += L
        for k in sorted(tot): print(k, tot[k])
    import lightgbm as lgb, frag_link
    rows = []
    for f in glob.glob(str(OUT / '*.json')): rows += json.load(open(f))
    X = np.array([[(-1 if r.get(k) is None else r.get(k, -1)) for k in frag_link.FEATS] for r in rows], dtype=np.float32)
    sets = np.array([r['set'] for r in rows]); y = np.array([r['lab'] == 'P' for r in rows], dtype=int)
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=0, num_threads=8)
    for tag, tr in [('A', ['t127a', 'audit32']), ('B', ['t127b', 'audit32']), ('full', ['t127a', 't127b', 'audit32'])]:
        m = np.isin(sets, tr)
        b = lgb.train(params, lgb.Dataset(X[m], y[m]), 400)
        json.dump(b.dump_model(), open('/workspace/cl/frag_lgb_%s.json' % tag, 'w'))
        p = b.predict(X)
        def auc(pp, yy):
            o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
            return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)
        te = {'A': 't127b', 'B': 't127a', 'full': 'hold36'}[tag]
        mm = sets == te
        line = 'model %s trained %s -> %s AUC %.3f' % (tag, tr, te, auc(p[mm], y[mm]))
        for th in [0.1, 0.2, 0.3, 0.5]:
            k = mm & (p >= th); line += ' | th %.1f chains %d P %d tp %d nodes %d' % (th, k.sum(), y[k].sum(), sum(rows[i]['tp'] for i in np.where(k)[0]), sum(rows[i]['L'] for i in np.where(k)[0]))
        print(line)
