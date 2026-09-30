"""Apply dfork.resolve with larger K as a post-step on P8 output graphs and score with the official metric.
usage: dfk_eval.py <tag> <graph_dir> '<json list of K>'"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds')
from pathlib import Path
from multiprocessing import Pool
import numpy as np


def job(a):
    f, K = a
    import evalx, dfork
    nodes, edges = evalx.load_graph_json(f)
    st = {}
    if K:
        edges, st = dfork.resolve(nodes, edges, K=K)
    r = evalx.score_movie(Path(f).stem, nodes, edges); r['K'] = K; r['rm'] = st.get('dfork_removed', 0); return r


if __name__ == '__main__':
    tag, gdir, Ks = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    fs = sorted(glob.glob(gdir + '/*.json'))
    with Pool(96) as p: R = p.map(job, [(f, K) for K in [0] + Ks for f in fs])
    from tracking_cellmot.metrics import summarise
    for K in [0] + Ks:
        rows = [r for r in R if r['K'] == K]; s = summarise(rows)
        print('%-8s K %2d score %.6f adjE %.6f divJ %.4f div %d/%d/%d removed %d' % (tag, K, s['score'], s['adj_edge_jaccard'], s['division_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'], sum(r['rm'] for r in rows)))
