"""Generic evaluation of a parameter-free graph rule with the official metric on all 199 movies, per embryo, with movie bootstrap.
usage: rule_eval.py <src: b5|p13> <module> '<json list of kwargs dicts>'"""
import os, sys, json, glob, importlib
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_k, '1')  # the container has pids.max 16896; unlimited BLAS/polars threads x pool workers deadlock it
sys.path.insert(0, '/workspace/cl/nm'); sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
SRC = {'b5': {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
              'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
              't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'},
       'p13': {s: '/workspace/cl/ps_p13_%s/graphs' % s for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']},
       'p14': {s: '/workspace/cl/ps_p14_%s/graphs' % s for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']},
       'p15': {s: '/workspace/cl/ps_p15_%s/graphs' % s for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']}}


FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def job(args):
    mod, f, V = args
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False  # blosc otherwise starts one thread per core in every worker (zarr reads)
    except Exception:
        pass
    import evalx
    m = importlib.import_module(mod)
    name = Path(f).stem
    s = [k for k in FULL if ('_%s/' % k) in f or ('b5f_%s/' % k) in f][0]
    nodes, edges = evalx.load_graph_json(f)
    res = []
    for i, kw in enumerate(V):
        if kw is None: nn, ne, st = nodes, edges, {}
        else:
            if getattr(m, 'WANTS_META', False):  # module may read the pre-ILP candidate graph / image of this movie
                kw = dict(kw, name=name, set=s, fullgeff=FULL[s] + '/' + name + '.geff', zarr='/workspace/data/train/' + name + '.zarr')
            nn, ne, st = m.apply(nodes, edges, **kw)
        r = evalx.score_movie(name, nn, ne); r.update(st); r['vi'] = i; r['set'] = s; res.append(r)
    return res


if __name__ == '__main__':
    src, mod = sys.argv[1], sys.argv[2]; V = [None] + json.loads(sys.argv[3])
    jobs = [(mod, f, V) for s, d in SRC[src].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(int(os.environ.get('RULE_POOL', '24')), maxtasksperchild=4) as p: R = [r for rs in p.map(job, jobs, chunksize=1) for r in rs]
    json.dump(R, open('/workspace/cl/nm/ver_rows_%s.json' % os.environ.get('VTAG', mod), 'w'))
    from tracking_cellmot.metrics import summarise
    rng = np.random.default_rng(0)
    base = {r['movie']: r for r in R if r['vi'] == 0}
    ms = sorted(base); K = [rng.integers(0, len(ms), len(ms)) for _ in range(300)]
    for i, kw in enumerate(V[1:], 1):
        cur = {r['movie']: r for r in R if r['vi'] == i}
        line = '%-40s' % json.dumps(kw)
        for emb in ['44b6', '6bba', '']:
            mm = [m for m in ms if m.startswith(emb)]
            a = summarise([base[m] for m in mm]); b = summarise([cur[m] for m in mm])
            line += ' | %s %+.5f (e %+.5f, div %+d/%+d)' % (emb or 'all', b['score'] - a['score'], b['adj_edge_jaccard'] - a['adj_edge_jaccard'],
                                                         b['division_tp'] - a['division_tp'], b['division_fp'] - a['division_fp'])
        bs = [summarise([cur[ms[j]] for j in k])['score'] - summarise([base[ms[j]] for j in k])['score'] for k in K]
        line += ' | CI [%+.5f, %+.5f]' % (np.quantile(bs, .025), np.quantile(bs, .975))
        cm = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]  # never seen by b1 or the link models
        line += ' | clean40 %+.5f' % (summarise([cur[m] for m in cm])['score'] - summarise([base[m] for m in cm])['score'])
        line += ' | nodes %+d' % (sum(cur[m]['num_pred_nodes'] for m in ms) - sum(base[m]['num_pred_nodes'] for m in ms))
        print(line, flush=True)
