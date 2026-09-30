"""Grid-evaluate sadd.apply (standalone chain insertion) with the official metric.
usage: sadd_eval.py <graph_dir> <full_dir> <tag> '<json cfg list>'"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from multiprocessing import Pool
import numpy as np


def job(a):
    f, fdir, ci, cfg = a
    import evalx, sadd, nswap
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    st = {}
    if cfg is not None:
        nodes, edges, st = sadd.apply(nodes, edges, nswap.load_full(Path(fdir) / (name + '.geff')), **cfg)
    r = evalx.score_movie(name, nodes, edges, detail=True); r['ci'] = ci; r.update(st); return r


if __name__ == '__main__':
    gdir, fdir, tag, cfgs = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    cfgs = [None] + cfgs
    fs = sorted(glob.glob(gdir + '/*.json'))
    with Pool(96) as p: R = p.map(job, [(f, fdir, ci, c) for ci, c in enumerate(cfgs) for f in fs])
    from tracking_cellmot.metrics import summarise
    base = {r['movie']: r for r in R if r['ci'] == 0}
    for ci, c in enumerate(cfgs):
        rows = [r for r in R if r['ci'] == ci]; s = summarise(rows)
        et = [sum(r[k] for r in rows) for k in ('edge_tp', 'edge_fp', 'edge_fn')]
        d = np.array([r['adj_edge_jaccard'] - base[r['movie']]['adj_edge_jaccard'] for r in rows])
        print('%-55s score %.6f adjE %.6f E %d/%d/%d div %d/%d/%d add %d chains %d nodes up/down %d/%d' % (
            json.dumps(c)[:55], s['score'], s['adj_edge_jaccard'], *et, s['division_tp'], s['division_fp'], s['division_fn'],
            sum(r.get('sadd_chains', 0) for r in rows), sum(r.get('sadd_nodes', 0) for r in rows), (d > 1e-9).sum(), (d < -1e-9).sum()), flush=True)
