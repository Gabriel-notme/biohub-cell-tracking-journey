"""Read-only: does a fork daughter have a SEPARATE pre-division history in the pre-ILP detections?
For fork p(t)->{a,b} and daughter k, look back in the full detection set: at frame t-j (j=0..3) the nearest detection to the
previous one (starting from k, step <= STEP um) that stays >= SEP um away from p's own lineage node at that frame.
hist_sep(k) = number of consecutive back-frames for which such a separate detection chain exists. A true daughter should have ~0
(before division it IS the mother, except a 1-2 frame pre-split); a stolen neighbour should have a long separate history."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
A = None


def job(a):
    s, name, forks = a
    import zarr
    fg = zarr.open_group(FULL[s] + '/' + name + '.geff', mode='r')
    fT = np.asarray(fg['nodes/props/t/values'][:]).astype(int)
    fP = np.stack([np.asarray(fg['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1) * S
    trees = {}
    byt = defaultdict(list)
    for i, t in enumerate(fT.tolist()): byt[t].append(i)
    for t, ix in byt.items(): trees[t] = (np.array(ix), cKDTree(fP[ix]))
    d = json.load(open('/workspace/cl/ps_p13_%s/graphs/%s.json' % (s, name)))
    nodes = {int(k): v for k, v in d['nodes'].items()}
    par = {int(e['target_id']): int(e['source_id']) for e in d['edges']}
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    out = []
    for r in forks:
        p, t = r['p'], r['t']
        plin = {}; x = p; tt = t
        while True:
            plin[tt] = pos[x]
            if x not in par or tt <= t - 4: break
            x = par[x]; tt -= 1
        res = {}
        for STEP, SEP in [(4.0, 4.0), (5.0, 5.0), (4.0, 6.0)]:
            hs = []
            for k in (r['a'], r['b']):
                cur = pos[k]; h = 0
                for j in range(0, 4):
                    tt = t - j
                    if tt not in trees or tt not in plin: break
                    ix, tr = trees[tt]
                    cand = [i for i in tr.query_ball_point(cur, STEP) if np.linalg.norm(fP[ix[i]] - plin[tt]) >= SEP]
                    if not cand: break
                    best = min(cand, key=lambda i: np.linalg.norm(fP[ix[i]] - cur))
                    cur = fP[ix[best]]; h += 1
                hs.append(h)
            res['h_%g_%g' % (STEP, SEP)] = hs
        out.append(dict(r, **res))
    return out


if __name__ == '__main__':
    R = json.load(open('/workspace/cl/ideas/div_lens_rows.json'))
    F = [r for r in R if r['kind'] == 'fork']
    by = defaultdict(list)
    for r in F: by[(r['set'], r['movie'])].append({k: r[k] for k in ('p', 'a', 'b', 't', 'lab', 'origin', 'kid_match', 'movie', 'set')})
    with Pool(16) as p: O = [x for xs in p.map(job, [(s, m, fs) for (s, m), fs in by.items()], chunksize=1) for x in xs]
    json.dump(O, open('/workspace/cl/ideas/dl_prehist_rows.json', 'w'))
    for key in ['h_4_4', 'h_5_5', 'h_4_6']:
        print('==', key)
        for thr in [1, 2, 3, 4]:
            c = Counter((r['lab'], r['origin']) for r in O if max(r[key]) >= thr)
            print('  max-daughter sep-history >= %d:' % thr, sorted(c.items()))
        # for FP: the unmatched daughter's value; for TP both
        fpu = [r[key][r['kid_match'].index(False)] for r in O if r['lab'] == 'FP' and False in r['kid_match']]
        print('  FP unmatched-daughter hist dist', Counter(fpu))
        print('  TP min/max daughter hist', Counter((min(r[key]), max(r[key])) for r in O if r['lab'] == 'TP'))
