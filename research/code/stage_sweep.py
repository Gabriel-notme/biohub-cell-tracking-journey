import os, sys, json, glob, time, copy
from pathlib import Path
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'
# usage: stage_sweep.py <start_dump_dir> <variants.json> <out.json> <nproc> [data_root]
gdir, vpath, outp = sys.argv[1], sys.argv[2], sys.argv[3]
nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 6
DATA = sys.argv[5] if len(sys.argv) > 5 else '/workspace/data/train'
VARIANTS = json.load(open(vpath))
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
    os.environ['POLARS_MAX_THREADS'] = '2'
_cache = {}
def get_stage(kind, cfg_over, model_over=None):
    sys.path.insert(0, ART)
    from lineage_pipeline import refiner_class
    sel = json.load(open(ART + '/final_selection.json'))['B5']
    base = {s['kind']: s for s in sel['stages']}[kind]
    cfg = dict(base['config']); cfg.update(cfg_over or {})
    files = model_over or base['model_files']
    key = json.dumps([kind, cfg, files], sort_keys=True)
    if key not in _cache:
        _cache[key] = refiner_class(kind)([Path(ART) / n for n in files], DATA, cfg, Path('/workspace/cache/sweep_' + kind))
    return _cache[key]
def job(path):
    import evalx
    name = Path(path).stem
    nodes0, edges0 = evalx.load_graph_json(path)
    rows = []
    for vname, steps in VARIANTS.items():
        nodes = {k: dict(v) for k, v in nodes0.items()}; edges = [dict(e) for e in edges0]
        t0 = time.time(); reps = []
        try:
            for st in steps:
                obj = get_stage(st['kind'], st.get('config'), st.get('models'))
                nodes, edges, rep = obj.refine(name, nodes, edges); reps.append({k: v for k, v in rep.items() if isinstance(v, (int, float))})
            r = evalx.score_movie(name, nodes, edges)
            if os.environ.get('SWEEP_SAVE'):
                d = Path(os.environ['SWEEP_SAVE']) / vname; d.mkdir(parents=True, exist_ok=True)
                (d / (name + '.json')).write_text(json.dumps({'nodes': {str(k): v for k, v in nodes.items()}, 'edges': edges}))
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {'error': repr(e), 'movie': name}
        r['cfg'] = vname; r['sec'] = time.time() - t0; r['reps'] = reps; rows.append(r)
    return rows
if __name__ == '__main__':
    paths = sorted(glob.glob(gdir + '/*.json'))
    lim = os.environ.get('SWEEP_LIMIT')
    if lim: paths = paths[:int(lim)]
    with get_context('spawn').Pool(nproc, initializer=init, initargs=([0, 1],)) as pool:
        allrows = [r for rs in pool.map(job, paths, chunksize=1) for r in rs]
    json.dump(allrows, open(outp, 'w'))
    from tracking_cellmot.metrics import summarise
    ref = list(VARIANTS)[0]
    import numpy as np
    by = {}
    for r in allrows:
        if 'error' in r: print('ERR', r['movie'], r['cfg'], r['error']); continue
        by.setdefault(r['cfg'], {})[r['movie']] = r
    base = summarise(list(by[ref].values()))
    for k in VARIANTS:
        rr = list(by.get(k, {}).values()); s = summarise(rr)
        print('%-28s score %.6f d=%+.6f edgeJ %.5f adj %.5f div %d/%d/%d nodes %d sec %.0f' % (k, s['score'], s['score'] - base['score'], s['edge_jaccard'], s['adj_edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'], sum(r['num_pred_nodes'] for r in rr), sum(r.get('sec', 0) for r in rr)), flush=True)
