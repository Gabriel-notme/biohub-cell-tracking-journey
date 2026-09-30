"""Per-movie B5 vs P11 on all 199 movies: which movies does the P-stage make worse, and is there a pattern (embryo, density, set)?"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def job(a):
    s, f = a
    import evalx
    nodes, edges = evalx.load_graph_json(f)
    r = evalx.score_movie(Path(f).stem, nodes, edges); r['set'] = s; return r


if __name__ == '__main__':
    jobs = [(s, f) for s, d in B5.items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(96) as p: R = p.map(job, jobs)
    json.dump(R, open('/workspace/cl/rev/b5all.json', 'w'))
    b5 = {r['movie']: r for r in R}
    p11 = {}
    for s in B5: p11.update({r['movie']: dict(r, set=s) for r in json.load(open('/workspace/cl/rev/p11_%s.json' % s))})
    from tracking_cellmot.metrics import summarise
    print('B5 all199 %.6f  P11 all199 %.6f' % (summarise(list(b5.values()))['score'], summarise([p11[m] for m in b5])['score']))
    for e in ['44b6', '6bba']:
        ms = [m for m in b5 if m.startswith(e)]
        print(e, 'B5 %.6f P11 %.6f' % (summarise([b5[m] for m in ms])['score'], summarise([p11[m] for m in ms])['score']))
    rows = []
    for m in b5:
        a, b = b5[m], p11[m]
        w = b['edge_tp'] + b['edge_fp'] + b['edge_fn']
        rows.append((m, a['set'] if 'set' in a else '', b['adj_edge_jaccard'] - a['adj_edge_jaccard'], w, a['num_pred_nodes'], b['division_tp'] - a['division_tp'], b['division_fp'] - a['division_fp']))
    rows.sort(key=lambda r: r[2] * r[3])
    print('worst weighted adjE changes (movie, set, d_adjE, weight, npred, dDivTP, dDivFP):')
    for r in rows[:15]: print('  ', r[0], r[1], '%+.4f' % r[2], r[3], r[4], r[5], r[6])
    d = np.array([r[2] for r in rows]); npred = np.array([r[4] for r in rows])
    for lo, hi in [(0, 10000), (10000, 25000), (25000, 40000), (40000, 1e9)]:
        k = (npred >= lo) & (npred < hi); print('npred %6d-%6d: n %3d mean d_adjE %+.4f  frac worse %.2f' % (lo, min(hi, 99999), k.sum(), d[k].mean(), (d[k] < 0).mean()))
    dtp = sum(r[5] for r in rows); dfp = sum(r[6] for r in rows); print('division change B5->P11: TP %+d FP %+d' % (dtp, dfp))
