"""Cross-embryo check: train edge_link/relink on one embryo's training movies (t127a+t127b+audit32), evaluate the P5 steps
on the OTHER embryo's clean validation movies (hold36+prev4), versus unmodified P3."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4'}


def train(emb):
    import lightgbm as lgb, edge_link, relink
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=8)
    out = {}
    for name, feats, cdir in [('edge', edge_link.FEATS, '/workspace/cl/ecands'), ('relink', relink.FEATS, '/workspace/cl/rcands')]:
        rows = []
        for s in ['t127a', 't127b', 'audit32']:
            for f in glob.glob('%s/%s__%s_*.json' % (cdir, s, emb)): rows += json.load(open(f))
        rows = [r for r in rows if r['lab'] != 'U']
        X = np.array([[(-1 if r.get(k) is None else r.get(k, -1)) for k in feats] for r in rows], dtype=np.float32)
        y = np.array([r['lab'] == 'P' for r in rows], dtype=int)
        b = lgb.train(params, lgb.Dataset(X, y), 400)
        p = '/workspace/cl/%s_lgb_emb%s.json' % (name, emb); json.dump(b.dump_model(), open(p, 'w')); out[name] = p
        print('trained', name, 'on', emb, 'n', len(rows), 'pos', int(y.sum()), flush=True)
    return out


def job(a):
    s, name, models = a
    import evalx, edge_link, relink
    nodes, edges = evalx.load_graph_json(Path('/workspace/cl/ps_p3_%s/graphs' % s) / (name + '.json'))
    r0 = evalx.score_movie(name, nodes, edges); r0['movie'] = name
    fp = Path(FULL[s]) / (name + '.geff')
    n2, e2, _ = relink.apply(nodes, edges, edge_link.load_full(fp), models['relink'], th=0.65)
    n2, e2, _ = edge_link.apply(n2, e2, fp, models['edge'], th=0.4)
    r1 = evalx.score_movie(name, n2, e2); r1['movie'] = name
    return r0, r1


if __name__ == '__main__':
    from tracking_cellmot.metrics import summarise
    M = {e: train(e) for e in ['44b6', '6bba']}
    rng = np.random.default_rng(0)
    for tr_emb, te_emb in [('44b6', '6bba'), ('6bba', '44b6')]:
        jobs = [(s, n, M[tr_emb]) for s, lst in [('hold36', '/workspace/hold36.txt'), ('prev4', '/workspace/preview4.txt')] for n in [l.strip() for l in open(lst) if l.strip()] if n.startswith(te_emb)]
        with Pool(24) as pool: res = pool.map(job, jobs)
        A = [a for a, b in res]; B = [b for a, b in res]
        sa, sb = summarise(A), summarise(B)
        out = []
        for _ in range(2000):
            k = rng.integers(0, len(A), len(A)); out.append(summarise([B[i] for i in k])['score'] - summarise([A[i] for i in k])['score'])
        out = np.array(out)
        up = sum(b['adj_edge_jaccard'] > a['adj_edge_jaccard'] + 1e-9 for a, b in res); dn = sum(b['adj_edge_jaccard'] < a['adj_edge_jaccard'] - 1e-9 for a, b in res)
        print('train %s -> test %s (n=%d clean movies): P3 %.6f -> %.6f  d %+.6f  CI [%+.5f, %+.5f] P>0 %.3f  movies up/down %d/%d' % (
            tr_emb, te_emb, len(A), sa['score'], sb['score'], sb['score'] - sa['score'], np.quantile(out, .025), np.quantile(out, .975), (out > 0).mean(), up, dn), flush=True)
