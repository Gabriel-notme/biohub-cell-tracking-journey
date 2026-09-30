import os, sys, json, glob, time
from pathlib import Path
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'
gdir, outdir = sys.argv[1], Path(sys.argv[2]); nproc = int(sys.argv[3]); grid = json.loads(sys.argv[4])
outdir.mkdir(parents=True, exist_ok=True)
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
def job(path):
    sys.path.insert(0, ART)
    import evalx, fork_veto
    from refine_events import EventRefiner
    name = Path(path).stem; cp = outdir / (name + '.forks.json')
    nodes, edges = evalx.load_graph_json(path)
    if cp.exists(): forks = json.loads(cp.read_text())
    else:
        er = EventRefiner([ART + '/b1_best.pt'], '/workspace/data/train', {'fork_ensemble': 'first'}, '/workspace/cache/veto_' + name)
        forks = fork_veto.fork_probs(er, name, nodes, edges); cp.write_text(json.dumps(forks))
    rows = [dict(evalx.score_movie(name, nodes, edges), cfg=-1)]
    for ci, cfg in enumerate(grid):
        ne, st = fork_veto.apply_veto(nodes, edges, forks, **cfg)
        r = evalx.score_movie(name, nodes, ne); r['cfg'] = ci; r['mod_stats'] = st; rows.append(r)
    return rows
if __name__ == '__main__':
    paths = sorted(glob.glob(gdir + '/*.json'))
    with get_context('spawn').Pool(nproc, initializer=init, initargs=([0, 1],)) as pool:
        allrows = [r for rs in pool.map(job, paths, chunksize=1) for r in rs]
    from tracking_cellmot.metrics import summarise
    base = summarise([r for r in allrows if r['cfg'] == -1])['score']
    for ci in range(-1, len(grid)):
        rr = [r for r in allrows if r['cfg'] == ci]; s = summarise(rr); tot = sum((r.get('mod_stats') or {}).get('vetoed', 0) for r in rr)
        print(ci, 'score %.6f d=%+.6f edgeJ %.5f div %d/%d/%d' % (s['score'], s['score'] - base, s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']), grid[ci] if ci >= 0 else 'base', 'vetoed', tot, flush=True)
