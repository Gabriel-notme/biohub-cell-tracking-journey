import os, sys, json, glob
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/p12ds')
os.environ['POLARS_MAX_THREADS'] = '2'
gdir = sys.argv[1]; LS = [int(x) for x in sys.argv[2].split(',')]
tag = sys.argv[3] if len(sys.argv) > 3 else Path(gdir).name
def job(a):
    p, L = a
    import evalx, prune
    name = Path(p).stem
    nodes, edges = evalx.load_graph_json(p)
    if L > 1: nodes, edges, _ = prune.prune_fragments(nodes, edges, L)
    r = evalx.score_movie(name, nodes, edges); r['L'] = L
    return r
if __name__ == '__main__':
    ps = sorted(glob.glob(gdir + '/*.json'))
    with Pool(32) as pool: rows = pool.map(job, [(p, L) for L in LS for p in ps], chunksize=1)
    from tracking_cellmot.metrics import summarise
    json.dump(rows, open('/workspace/runs/prune_%s.json' % tag, 'w'))
    for L in LS:
        rr = [r for r in rows if r['L'] == L]; s = summarise(rr)
        print('%s L=%d n=%d score %.6f adjE %.6f E %.6f div %d/%d/%d nodes %d' % (tag, L, len(rr), s['score'], s['adj_edge_jaccard'], s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'], sum(r['num_pred_nodes'] for r in rr)), flush=True)
