"""Evaluate shadow.apply variants with the official metric on a graph source for all 199 movies.
usage: shadow_eval.py <src: b5|p13> <variants json list of [D,L,mode]> -> per-movie rows cached in /workspace/cl/sh/<src>_<tag>.json"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '1')
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
SRC = {'b5': {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
              'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
              't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'},
       'p13': {s: '/workspace/cl/ps_p13_%s/graphs' % s for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']}}
OUT = Path('/workspace/cl/sh'); OUT.mkdir(exist_ok=True)


def job(args):
    src, s, f, V = args
    import evalx, shadow
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    res = []
    for v in V:
        if v is None: nn, ne, st = nodes, edges, {}
        else:
            m = None
            if v[2] == 'oracle':
                from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
                gt, _ = evalx.load_gt(name); pred, mapping = evalx.to_graph(nodes, edges); inv = {b: a for a, b in mapping.items()}
                ma = _matched_node_attrs(_match_full(pred, gt, evalx.SCALE, 7.)); m = {inv[int(a)] for a in ma[evalx.K.NODE_ID].to_list()}
            nn, ne, st = shadow.apply(nodes, edges, D=v[0], L=v[1], mode=v[2], matched=m)
        r = evalx.score_movie(name, nn, ne); r.update(st); r['set'] = s; r['v'] = v; res.append(r)
    return res


if __name__ == '__main__':
    src = sys.argv[1]; V = [None] + [tuple(v) for v in json.loads(sys.argv[2])]
    jobs = [(src, s, f, V) for s, d in SRC[src].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(96) as p: R = [r for rs in p.map(job, jobs, chunksize=1) for r in rs]
    tag = '_'.join('%s-%s-%s' % v for v in V[1:])
    json.dump(R, open(OUT / ('%s_%s.json' % (src, tag)), 'w'))
    from tracking_cellmot.metrics import summarise
    rng = np.random.default_rng(0)
    base = {r['movie']: r for r in R if r['v'] is None}
    for v in V[1:]:
        cur = {r['movie']: r for r in R if r['v'] is not None and tuple(r['v']) == v}
        line = 'D=%.1f L=%d %-7s' % v
        for emb in ['44b6', '6bba', '']:
            ms = sorted(m for m in base if m.startswith(emb))
            a = summarise([base[m] for m in ms]); b = summarise([cur[m] for m in ms])
            line += ' | %s %+.5f (e %+.5f, div %+d/%+d)' % (emb or 'all', b['score'] - a['score'], b['adj_edge_jaccard'] - a['adj_edge_jaccard'],
                                                         b['division_tp'] - a['division_tp'], b['division_fp'] - a['division_fp'])
        ms = sorted(base); bs = []
        for _ in range(300):
            k = rng.integers(0, len(ms), len(ms)); bs.append(summarise([cur[ms[i]] for i in k])['score'] - summarise([base[ms[i]] for i in k])['score'])
        line += ' | CI [%+.5f, %+.5f] nodes_rm %d' % (np.quantile(bs, .025), np.quantile(bs, .975), sum(cur[m].get('shadow_nodes', 0) for m in ms))
        print(line, flush=True)
