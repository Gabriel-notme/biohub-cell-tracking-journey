"""div_diff.py <tagA> <tagB>: forks present in B but not A (and vice versa) on the full P-stage graphs (/workspace/cl/p21/ps_<tag>_<set>),
labelled with the official division scorer (tp / fp / ne = not evaluable), plus official GT division TP/FP/FN per group."""
import os, sys, json, glob
for v in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'): os.environ[v] = '1'
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from multiprocessing import Pool
from collections import Counter
import numpy as np
S = np.array([1.625, 0.40625, 0.40625]); SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
A, B = sys.argv[1], sys.argv[2]

def lab(name, f):
    import evalx
    from tracking_cellmot.division_metrics import score_divisions
    nodes, edges = evalx.load_graph_json(f)
    g, mp = evalx.to_graph(nodes, edges, rounding=True); inv = {v: k for k, v in mp.items()}
    gt, _ = evalx.load_gt(name)
    ds = score_divisions(g, gt, scale=tuple(S), max_distance=7.0)
    ch = {}
    for e in edges: ch.setdefault(int(e['source_id']), []).append(int(e['target_id']))
    tp = {inv[x] for x in ds.tp_forks}; fp = {inv[x] for x in ds.fp_forks}
    out = {}
    for p, c in ch.items():
        if len(c) == 2:
            v = nodes[p]; key = (int(v['t']), round(float(v['z']), 2), round(float(v['y']), 2), round(float(v['x']), 2))
            out[key] = 'tp' if p in tp else ('fp' if p in fp else 'ne')
    return out, sum(ds.scores.values()), len(ds.scores)

def job(a):
    s, name = a
    fa = '/workspace/cl/p21/ps_%s_%s/graphs/%s.json' % (A, s, name); fb = '/workspace/cl/p21/ps_%s_%s/graphs/%s.json' % (B, s, name)
    la, tpa, ng = lab(name, fa); lb, tpb, _ = lab(name, fb)
    add = [lb[k] for k in lb if k not in la]; rem = [la[k] for k in la if k not in lb]
    chg = [(la[k], lb[k]) for k in la if k in lb and la[k] != lb[k]]
    return dict(movie=name, set=s, add=add, rem=rem, chg=chg, gt_tp_a=tpa, gt_tp_b=tpb, ngt=ng)

if __name__ == '__main__':
    todo = [(s, os.path.basename(f)[:-5]) for s in SETS for f in sorted(glob.glob('/workspace/cl/p21/ps_%s_%s/graphs/*.json' % (A, s)))]
    with Pool(60) as p: R = p.map(job, todo, chunksize=1)
    json.dump(R, open('/workspace/cl/p21/divdiff_%s_%s.json' % (A, B), 'w'))
    for g, sel in (('all', lambda r: True), ('clean40', lambda r: r['set'] in ('hold36', 'prev4')), ('insample', lambda r: r['set'] not in ('hold36', 'prev4')), ('44b6', lambda r: r['movie'].startswith('44b6')), ('6bba', lambda r: r['movie'].startswith('6bba'))):
        rr = [r for r in R if sel(r)]
        print('%-8s forks added %s | removed %s | relabelled %s | GT div TP %d -> %d (of %d) | movies with change %d' % (g, dict(Counter(x for r in rr for x in r['add'])), dict(Counter(x for r in rr for x in r['rem'])),
              dict(Counter(x for r in rr for x in r['chg'])), sum(r['gt_tp_a'] for r in rr), sum(r['gt_tp_b'] for r in rr), sum(r['ngt'] for r in rr), sum(1 for r in rr if r['add'] or r['rem'])))
