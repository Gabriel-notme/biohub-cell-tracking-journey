import os, sys, json, glob, time
from pathlib import Path
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'
gdir, outdir = sys.argv[1], Path(sys.argv[2]); nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
grid = json.loads(sys.argv[4]) if len(sys.argv) > 4 else []
outdir.mkdir(parents=True, exist_ok=True)

def init(gpu_list):
    import multiprocessing as mp
    idx = mp.current_process()._identity[0] - 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list[idx % len(gpu_list)])

def job(path):
    sys.path.insert(0, ART)
    os.environ['BIOHUB_ART'] = ART
    import evalx, div_complete_b1 as div_complete
    from refine_events import EventRefiner
    name = Path(path).stem
    cp = outdir / (name + '.cands.json')
    CK = json.loads(os.environ.get('CAND_KW', '{}'))
    nodes, edges = evalx.load_graph_json(path)
    if cp.exists(): cands = json.loads(cp.read_text())
    else:
        er = EventRefiner([ART + '/b1_best.pt'], '/workspace/data/train', {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, '/workspace/cache/event_' + name)
        cands = div_complete.score_candidates2(er, name, nodes, edges, **CK)
        cp.write_text(json.dumps(cands))
    rows = []
    base = evalx.score_movie(name, nodes, edges); base['cfg'] = -1; rows.append(base)
    for ci, cfg in enumerate(grid):
        ne, st = div_complete.apply(nodes, edges, cands, **{k: v for k, v in cfg.items() if k != 'prune_min_len'})
        nn = nodes
        if cfg.get('prune_min_len'):
            import prune
            nn, ne, pst = prune.prune_fragments(nodes, ne, cfg['prune_min_len']); st = dict(st, **pst)
        r = evalx.score_movie(name, nn, ne); r['cfg'] = ci; r['mod_stats'] = st; rows.append(r)
    return rows

if __name__ == '__main__':
    paths = sorted(glob.glob(gdir + '/*.json'))
    t0 = time.time()
    with get_context('spawn').Pool(nproc, initializer=init, initargs=([0, 1],)) as pool:
        allrows = [r for rs in pool.map(job, paths, chunksize=1) for r in rs]
    from tracking_cellmot.metrics import summarise
    base = summarise([r for r in allrows if r['cfg'] == -1])['score']
    out = {}
    for ci in range(-1, len(grid)):
        rr = [r for r in allrows if r['cfg'] == ci]; s = summarise(rr)
        tot = {}
        for r in rr:
            for k, v in (r.get('mod_stats') or {}).items(): tot[k] = tot.get(k, 0) + v
        out[ci] = {'summary': s, 'cfg': grid[ci] if ci >= 0 else 'baseline', 'stats': tot}
        print(ci, 'score %.6f d=%+.6f edgeJ %.5f adj %.5f div %d/%d/%d' % (s['score'], s['score'] - base, s['edge_jaccard'], s['adj_edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']), json.dumps(out[ci]['cfg']), tot, flush=True)
    json.dump({'out': out, 'rows': allrows}, open(outdir / 'grid_result.json', 'w'))
    print('sec', round(time.time() - t0, 1))
    