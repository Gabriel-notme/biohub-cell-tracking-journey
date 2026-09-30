"""dfork K sweep on top of existing graphs. usage: dfk_test.py <graph_dir> <set_tag> <out_prefix> K1,K2,..."""
import os, sys, json, glob
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/p12ds')
os.environ['POLARS_MAX_THREADS'] = '2'
gdir, tag, pref, Ks = sys.argv[1], sys.argv[2], sys.argv[3], [int(k) for k in sys.argv[4].split(',')]
def job(args):
    p, K = args
    import evalx, dfork, prune
    nodes, edges = evalx.load_graph_json(p)
    ne, st = dfork.resolve(nodes, edges, K)
    nn, ne, st2 = prune.prune_fragments(nodes, ne, 2)
    r = evalx.score_movie(Path(p).stem, nn, ne)
    r['dfork_removed'] = st['dfork_removed']
    return K, r
if __name__ == '__main__':
    ps = sorted(glob.glob(gdir + '/*.json'))
    with Pool(32) as pool: res = pool.map(job, [(p, K) for K in Ks for p in ps])
    from tracking_cellmot.metrics import summarise
    for K in Ks:
        rows = [r for k, r in res if k == K]
        json.dump(rows, open('%s/%s__dfK%d.json' % (pref, tag, K), 'w'))
        s = summarise(rows)
        print('K', K, 'removed', sum(r['dfork_removed'] for r in rows), 'score %.6f' % s['score'], 'div %d/%d/%d' % (s['division_tp'], s['division_fp'], s['division_fn']), 'E %.6f' % s['edge_jaccard'])
