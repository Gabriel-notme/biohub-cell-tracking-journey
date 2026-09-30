"""Paired comparison of two graph dirs with the official metric + movie bootstrap.
usage: compare.py <base_dir> <var_dir> [tag] [nboot]"""
import os, sys, json, glob
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np
bdir, vdir = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else ''; NB = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
def job(p):
    import evalx
    nodes, edges = evalx.load_graph_json(p)
    return evalx.score_movie(Path(p).stem, nodes, edges)
if __name__ == '__main__':
    names = sorted(Path(p).stem for p in glob.glob(vdir + '/*.json'))
    names = [n for n in names if os.path.exists(bdir + '/' + n + '.json')]
    with Pool(32) as pool:
        B = pool.map(job, [bdir + '/' + n + '.json' for n in names]); V = pool.map(job, [vdir + '/' + n + '.json' for n in names])
    from tracking_cellmot.metrics import summarise
    sb, sv = summarise(B), summarise(V)
    print('%s n=%d base %.6f var %.6f delta %+.6f | edgeJ %.6f->%.6f adjE %.6f->%.6f div %d/%d/%d -> %d/%d/%d' % (tag, len(names), sb['score'], sv['score'], sv['score'] - sb['score'], sb['edge_jaccard'], sv['edge_jaccard'], sb['adj_edge_jaccard'], sv['adj_edge_jaccard'], sb['division_tp'], sb['division_fp'], sb['division_fn'], sv['division_tp'], sv['division_fp'], sv['division_fn']))
    for emb in ['44b6', '6bba']:
        ib = [r for r in B if r['movie'].startswith(emb)]; iv = [r for r in V if r['movie'].startswith(emb)]
        if ib:
            a, b = summarise(ib), summarise(iv)
            print('   %s n=%d delta %+.6f (base %.6f) div %d/%d/%d -> %d/%d/%d' % (emb, len(ib), b['score'] - a['score'], a['score'], a['division_tp'], a['division_fp'], a['division_fn'], b['division_tp'], b['division_fp'], b['division_fn']))
    rng = np.random.default_rng(0); d = []
    idx = np.arange(len(names))
    for _ in range(NB):
        s = rng.choice(idx, len(idx), replace=True)
        d.append(summarise([V[i] for i in s])['score'] - summarise([B[i] for i in s])['score'])
    d = np.array(d)
    print('   bootstrap delta mean %+.5f  95%% CI [%+.5f, %+.5f]  P(delta>0)=%.3f' % (d.mean(), np.quantile(d, .025), np.quantile(d, .975), (d > 0).mean()))
    ch = [(n, v['score'] if 'score' in v else None) for n, v in zip(names, V)]
    diffs = [(n, round((v['edge_tp'] - b['edge_tp']), 1), v['division_tp'] - b['division_tp'], v['division_fp'] - b['division_fp']) for n, b, v in zip(names, B, V) if (v['division_tp'], v['division_fp'], v['edge_tp'], v['edge_fp']) != (b['division_tp'], b['division_fp'], b['edge_tp'], b['edge_fp'])]
    print('   changed movies (edgeTPdelta, divTPdelta, divFPdelta):', diffs)
