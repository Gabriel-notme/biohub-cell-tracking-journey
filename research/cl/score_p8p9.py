"""Per-movie official metric for P8 vs P9 graphs (rounded like export); checks division counts unchanged and per-movie edge deltas."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool


def job(args):
    s, cfg, name = args
    import evalx
    nodes, edges = evalx.load_graph_json('/workspace/cl/ps_%s_%s/graphs/%s.json' % (cfg, s, name))
    r = evalx.score_movie(name, nodes, edges)
    return (s, cfg, name, r)


if __name__ == '__main__':
    sets = sys.argv[1].split(',')
    jobs = []
    for s in sets:
        names = sorted(p.stem for p in Path('/workspace/cl/ps_p8_%s/graphs' % s).glob('*.json'))
        for n in names:
            for c in ['p8', 'p9']: jobs.append((s, c, n))
    with Pool(12) as p: R = p.map(job, jobs)
    import evalx
    from tracking_cellmot.metrics import summarise
    res = {}
    for s, c, n, r in R: res[(s, c, n)] = r
    json.dump({'|'.join(k): v for k, v in res.items()}, open('/workspace/verify/codereview/score_p8p9_%s.json' % '_'.join(sets), 'w'), default=float)
    for s in sets:
        names = sorted({n for (ss, c, n) in res if ss == s})
        print('SET', s)
        for n in names:
            a = res[(s, 'p8', n)]; b = res[(s, 'p9', n)]
            dsame = (a['division_tp'], a['division_fp'], a['division_fn']) == (b['division_tp'], b['division_fp'], b['division_fn'])
            print('%-15s Ntot %7.0f Npred %6d->%6d TP %5d->%5d FP %5d->%5d FN %5d->%5d adjJ %.5f->%.5f (%+.5f) div(tp,fp,fn) %s %s' % (
                n, a['n_total'], a['num_pred_nodes'], b['num_pred_nodes'], a['edge_tp'], b['edge_tp'], a['edge_fp'], b['edge_fp'], a['edge_fn'], b['edge_fn'],
                a['adj_edge_jaccard'], b['adj_edge_jaccard'], b['adj_edge_jaccard'] - a['adj_edge_jaccard'],
                (a['division_tp'], a['division_fp'], a['division_fn']), 'SAME' if dsame else 'DIFF -> %s' % str((b['division_tp'], b['division_fp'], b['division_fn']))))
        for c in ['p8', 'p9']:
            sm = summarise([res[(s, c, n)] for n in names])
            print('  %s %s score %.6f adjJ %.6f divJ %.6f div(tp,fp,fn)=(%d,%d,%d)' % (s, c, sm['score'], sm['adj_edge_jaccard'], sm['division_jaccard'], sm['division_tp'], sm['division_fp'], sm['division_fn']))
