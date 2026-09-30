import os, sys, json, glob, importlib, time
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
# usage: test_mod.py <graph_dir> <module:function> <grid_json> <out_json> [nproc]
gdir, modfn, grid, outp = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 8
mod, fn = modfn.split(':')

def job(args):
    path, ci = args
    import evalx
    m = importlib.import_module(mod); f = getattr(m, fn)
    nodes, edges = evalx.load_graph_json(path)
    name = Path(path).stem
    if ci < 0: st = {}
    else:
        res = f(nodes, edges, **grid[ci])
        if len(res) == 3: nodes, edges, st = res
        else: edges, st = res
    row = evalx.score_movie(name, nodes, edges); row['cfg'] = ci; row['mod_stats'] = st
    return row

paths = sorted(glob.glob(gdir + '/*.json'))
jobs = [(p, ci) for ci in range(-1, len(grid)) for p in paths]
t0 = time.time()
with Pool(nproc) as pool: rows = pool.map(job, jobs, chunksize=1)
from tracking_cellmot.metrics import summarise
res = {}
for ci in range(-1, len(grid)):
    rr = [r for r in rows if r['cfg'] == ci]
    s = summarise(rr); s['cfg'] = grid[ci] if ci >= 0 else 'baseline'
    s['mod_stats_total'] = {}
    for r in rr:
        for k, v in (r['mod_stats'] or {}).items():
            if isinstance(v, (int, float)): s['mod_stats_total'][k] = s['mod_stats_total'].get(k, 0) + v
    res[ci] = s
base = res[-1]['score']
for ci in sorted(res):
    s = res[ci]
    print(ci, 'score %.6f d=%+.6f edgeJ %.5f adj %.5f div %d/%d/%d' % (s['score'], s['score'] - base, s['edge_jaccard'], s['adj_edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']), json.dumps(s['cfg']), s['mod_stats_total'])
json.dump({'res': res, 'rows': rows}, open(outp, 'w'))
print('movies', len(paths), 'sec', round(time.time() - t0, 1))
