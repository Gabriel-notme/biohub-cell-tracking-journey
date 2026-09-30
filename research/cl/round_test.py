"""Score a graph dir with integer-rounded coordinates (as exported) vs raw float coordinates."""
import os, sys, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool


def job(a):
    f, rnd = a
    import evalx
    nodes, edges = evalx.load_graph_json(f)
    r = evalx.score_movie(Path(f).stem, nodes, edges, rounding=rnd); r['rnd'] = rnd; return r


if __name__ == '__main__':
    fs = sorted(glob.glob(sys.argv[1] + '/*.json'))
    with Pool(72) as p: R = p.map(job, [(f, r) for r in [True, False] for f in fs])
    from tracking_cellmot.metrics import summarise
    for rnd in [True, False]:
        s = summarise([r for r in R if r['rnd'] == rnd])
        print('rounding', rnd, 'score %.6f adjE %.6f edgeJ %.6f div %d/%d/%d' % (s['score'], s['adj_edge_jaccard'], s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']))
