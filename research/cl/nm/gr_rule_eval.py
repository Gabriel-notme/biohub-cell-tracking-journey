"""Copy of /workspace/cl/rule_eval.py for the gr_ experiments: output under /workspace/cl/nm, pool <= 24, optional set filter.
usage: gr_rule_eval.py <src> <module> '<json list of kwargs>' [tag]   env GR_SETS=hold36,prev4 limits the sets"""
import os, sys, json, glob, importlib, time
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/cl/nm'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
SRC = {'p15': {s: '/workspace/cl/ps_p15_%s/graphs' % s for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']}}
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def job(args):
    mod, f, V = args
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx
    m = importlib.import_module(mod)
    name = Path(f).stem
    s = [k for k in FULL if ('_%s/' % k) in f][0]
    nodes, edges = evalx.load_graph_json(f)
    res = []
    for i, kw in enumerate(V):
        t0 = time.time()
        if kw is None: nn, ne, st = nodes, edges, {}
        else:
            if getattr(m, 'WANTS_META', False):
                kw = dict(kw, name=name, set=s, fullgeff=FULL[s] + '/' + name + '.geff', zarr='/workspace/data/train/' + name + '.zarr')
            nn, ne, st = m.apply(nodes, edges, **kw)
        dt = time.time() - t0
        r = evalx.score_movie(name, nn, ne); r.update(st); r['vi'] = i; r['set'] = s; r['sec'] = round(dt, 2); res.append(r)
    return res


if __name__ == '__main__':
    src, mod = sys.argv[1], sys.argv[2]; V = [None] + json.loads(sys.argv[3]); tag = sys.argv[4] if len(sys.argv) > 4 else mod
    sets = os.environ.get('GR_SETS', '').split(',') if os.environ.get('GR_SETS') else list(SRC[src])
    jobs = [(mod, f, V) for s, d in SRC[src].items() if s in sets for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(min(24, int(os.environ.get('RULE_POOL', '24'))), maxtasksperchild=4) as p: R = [r for rs in p.map(job, jobs, chunksize=1) for r in rs]
    json.dump(R, open('/workspace/cl/nm/gr_rows_%s.json' % tag, 'w'))
    from tracking_cellmot.metrics import summarise
    rng = np.random.default_rng(0)
    base = {r['movie']: r for r in R if r['vi'] == 0}
    ms = sorted(base); K = [rng.integers(0, len(ms), len(ms)) for _ in range(300)]
    for i, kw in enumerate(V[1:], 1):
        cur = {r['movie']: r for r in R if r['vi'] == i}
        line = '%-40s' % json.dumps(kw)
        for emb in ['44b6', '6bba', '']:
            mm = [m for m in ms if m.startswith(emb)]
            if not mm: continue
            a = summarise([base[m] for m in mm]); b = summarise([cur[m] for m in mm])
            line += ' | %s %+.5f (e %+.5f, tp %+d fp %+d, div %+d/%+d)' % (emb or 'all', b['score'] - a['score'], b['adj_edge_jaccard'] - a['adj_edge_jaccard'],
                    sum(cur[m]['edge_tp'] - base[m]['edge_tp'] for m in mm), sum(cur[m]['edge_fp'] - base[m]['edge_fp'] for m in mm),
                    b['division_tp'] - a['division_tp'], b['division_fp'] - a['division_fp'])
        bs = [summarise([cur[ms[j]] for j in k])['score'] - summarise([base[ms[j]] for j in k])['score'] for k in K]
        line += ' | CI [%+.5f, %+.5f]' % (np.quantile(bs, .025), np.quantile(bs, .975))
        cm = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]
        if cm: line += ' | clean40 %+.5f' % (summarise([cur[m] for m in cm])['score'] - summarise([base[m] for m in cm])['score'])
        line += ' | nodes %+d | sec/movie %.2f' % (sum(cur[m]['num_pred_nodes'] for m in ms) - sum(base[m]['num_pred_nodes'] for m in ms),
                                                    np.mean([cur[m]['sec'] for m in ms]))
        print(line, flush=True)
