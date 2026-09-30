"""early.py <half> <tagA> <tagB>: official score of B5-level lineage graphs for the movies of one half: original b5f, nbrun <tagA>, nbrun <tagB>."""
import os, sys, json, glob
for v in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'): os.environ[v] = '1'
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from multiprocessing import Pool
import numpy as np
H, A, B = sys.argv[1:4]
SETD = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32', 't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
LST = {'hold36': 'hold36', 'prev4': 'preview4', 'audit32': 'audit32', 't127a': 't127a', 't127b': 't127b'}
mset = {}
for s, l in LST.items():
    for m in open('/workspace/%s.txt' % l).read().split(): mset[m] = s
def job(m):
    import evalx
    out = {}
    for k, f in (('b5f', SETD[mset[m]] + '/working/lineage_graphs/%s.json' % m), (A, '/workspace/nbrun/%s_%s/lin/%s.json' % (A, H, m)), (B, '/workspace/nbrun/%s_%s/lin/%s.json' % (B, H, m))):
        if not os.path.exists(f): return None
        out[k] = evalx.score_movie(m, *evalx.load_graph_json(f))
    return m, out
if __name__ == '__main__':
    ms = sorted(os.path.basename(p)[:-5] for p in glob.glob('/workspace/nbrun/%s_%s/lin/*.json' % (A, H)))
    with Pool(60) as p: R = [r for r in p.map(job, ms) if r]
    from tracking_cellmot.metrics import summarise
    json.dump({m: o for m, o in R}, open('/workspace/nbrun/early_%s_%s_%s.json' % (H, A, B), 'w'))
    for g, sel in (('all', lambda m: True), ('44b6', lambda m: m.startswith('44b6')), ('6bba', lambda m: m.startswith('6bba')), ('clean40', lambda m: mset[m] in ('hold36', 'prev4')), ('unseen72', lambda m: mset[m] in ('hold36', 'prev4', 'audit32'))):
        rr = [(m, o) for m, o in R if sel(m)]
        s = {k: summarise([o[k] for m, o in rr]) for k in ('b5f', A, B)}
        print('%-8s n=%3d | b5f %.5f div %d/%d | %s %.5f div %d/%d | %s %.5f div %d/%d | %s-b5f %+.5f | %s-%s %+.5f' % (g, len(rr), s['b5f']['score'], s['b5f']['division_tp'], s['b5f']['division_fp'],
              A, s[A]['score'], s[A]['division_tp'], s[A]['division_fp'], B, s[B]['score'], s[B]['division_tp'], s[B]['division_fp'], A, s[A]['score'] - s['b5f']['score'], B, A, s[B]['score'] - s[A]['score']))
