"""Labels and features of long-range (14 < d <= R) second-pass edge_link links; per-movie score deltas.
usage: python3 ideas/pp_long_an.py R [cross]"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import rule_eval as RE
R = float(sys.argv[1]); MODEL = sys.argv[2] if len(sys.argv) > 2 else None
S = np.array([1.625, .40625, .40625])


def job(f):
    import evalx
    from ideas.pp_rmax_an import labels
    sys.path.insert(0, '/workspace/p56stage')
    from ideas import pp_pipe
    name = Path(f).stem; st = [k for k in RE.FULL if ('_%s/' % k) in f][0]
    nodes, edges = evalx.load_graph_json(f)
    n2, e2, s2 = pp_pipe.apply(nodes, edges, name=name, set=st, fullgeff=RE.FULL[st] + '/' + name + '.geff', long_r=R, long_model=MODEL, long_dump=True)
    rows = s2.get('long_rows', [])
    if not rows: return name, st, [], None, None
    gt, _ = evalx.load_gt(name)
    lab = labels(n2, e2, gt)
    for r in rows:
        r['lab'] = lab.get((r['s'], r['d']), 'pruned')
        a, b = n2[r['s']], n2[r['d']]
        r['dzum'] = round(abs(a['z'] - b['z']) * 1.625, 2)
    r0 = evalx.score_movie(name, nodes, edges); r1 = evalx.score_movie(name, n2, e2)
    return name, st, rows, r0, r1


if __name__ == '__main__':
    fs = [f for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(60) as p: res = p.map(job, fs, chunksize=1)
    allr = [dict(r, movie=m, set=s) for m, s, rows, _, _ in res for r in rows]
    json.dump(allr, open('/workspace/cl/ideas/pp_long_rows.json', 'w'))
    print('long links', len(allr), Counter((r['lab'], r['movie'][:4]) for r in allr))
    for lab in ['TP', 'FP', 'U']:
        x = [r for r in allr if r['lab'] == lab]
        if not x: continue
        for k in ['dist', 'dz', 'dxy', 'dpred', 'dpred_d', 'cos_s', 'cos_d', 'sp_s', 'sp_d', 'hist_s', 'fut_d', 'fe', 'n_s', 'n_d', 'rk_s', 'rk_d', 'nn_s_next', 'nn_d_prev', 'p', 'tt', 'z']:
            v = np.array([r[k] for r in x], float)
            print(lab, k, len(v), np.round(np.quantile(v, [0, .1, .25, .5, .75, .9, 1]), 2).tolist())
    from tracking_cellmot.metrics import summarise
    ch = [(m, s, r0, r1) for m, s, rows, r0, r1 in res if r0 is not None]
    d = sorted([(round(summarise([r1])['score'] - summarise([r0])['score'], 5), m, s, r1['edge_tp'] - r0['edge_tp'], r1['edge_fp'] - r0['edge_fp']) for m, s, r0, r1 in ch])
    print('per-movie deltas (movies with long links):', len(d), 'pos', sum(1 for x in d if x[0] > 0), 'neg', sum(1 for x in d if x[0] < 0))
    for x in d:
        if x[0] != 0: print(x)
