"""Grid-evaluate extend.apply on a directory of graphs with the official metric (per-movie rows -> summarise).
usage: ext_eval.py <graph_dir> <movie_list.txt> <tag> '<json list of configs>'"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from multiprocessing import Pool
import numpy as np


def job(args):
    gdir, name, ci, cfg = args
    import evalx, extend
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    st = {}
    if cfg is not None:
        nodes, edges, st = extend.apply(nodes, edges, **cfg)
    r = evalx.score_movie(name, nodes, edges, detail=True); r['movie'] = name; r['ci'] = ci; r.update(st)
    return r


if __name__ == '__main__':
    gdir, lst, tag, cfgs = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    names = [l.strip() for l in open(lst) if l.strip()]
    names = [n for n in names if (Path(gdir) / (n + '.json')).exists()]
    cfgs = [None] + cfgs
    jobs = [(gdir, n, ci, c) for ci, c in enumerate(cfgs) for n in names]
    with Pool(96) as p: R = p.map(job, jobs)
    from tracking_cellmot.metrics import summarise
    out = Path('/workspace/cl/ext'); out.mkdir(exist_ok=True)
    json.dump({'cfgs': cfgs, 'rows': R}, open(out / (tag + '.json'), 'w'))
    base = {r['movie']: r for r in R if r['ci'] == 0}
    for ci, c in enumerate(cfgs):
        rows = [r for r in R if r['ci'] == ci]
        s = summarise(rows)
        et = [sum(r[k] for r in rows) for k in ('edge_tp', 'edge_fp', 'edge_fn')]
        d = np.array([r['adj_edge_jaccard'] - base[r['movie']]['adj_edge_jaccard'] for r in rows])
        an = {}
        for r in rows:
            for k, v in r.get('analysis', {}).items(): an[k] = an.get(k, 0) + v
        print('%-70s score %.6f adjE %.6f E %d/%d/%d div %d/%d/%d ext %d/%d up/down %d/%d | fn_dst_missing %d fn_src_missing %d fp_other %d' % (
            json.dumps(c)[:70], s['score'], s['adj_edge_jaccard'], *et, s['division_tp'], s['division_fp'], s['division_fn'],
            sum(r.get('ext_fwd_nodes', 0) for r in rows), sum(r.get('ext_bwd_nodes', 0) for r in rows), (d > 1e-9).sum(), (d < -1e-9).sum(),
            an.get('fn_dst_missing', 0), an.get('fn_src_missing', 0), an.get('fp_to_other_gt', 0)), flush=True)
