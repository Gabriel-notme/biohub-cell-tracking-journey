"""P21 probe: timing offset between predicted forks and GT divisions (official matching), P15 graphs, 199 movies.
For each GT division: official TP/FN; nearest pred fork within |dt|<=5 frames and 12 um of the GT divider (rounded coords);
for each pred fork: official TP/FP/other. Writes div_timing.json."""
import os, sys, json, glob
os.environ['POLARS_MAX_THREADS'] = '1'
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'): os.environ[v] = '1'
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']

def job(a):
    s, f = a
    import evalx
    from tracking_cellmot.division_metrics import score_divisions
    name = os.path.basename(f)[:-5]
    nodes, edges = evalx.load_graph_json(f)
    g, mp = evalx.to_graph(nodes, edges, rounding=True)
    inv = {v: k for k, v in mp.items()}
    gt, _ = evalx.load_gt(name)
    ds = score_divisions(g, gt, scale=tuple(S), max_distance=7.0)
    ga = gt.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x']).to_dicts()
    gpos = {r['node_id']: (int(r['t']), np.array([r['z'], r['y'], r['x']]) * S) for r in ga}
    ch = {}; par = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch.setdefault(u, []).append(v); par[v] = u
    pos = {n: (int(v['t']), np.array([max(0, int(round(v[c]))) for c in 'zyx']) * S) for n, v in nodes.items()}
    forks = [n for n, c in ch.items() if len(c) == 2]
    tpf = {inv[x] for x in ds.tp_forks}; fpf = {inv[x] for x in ds.fp_forks}
    gdiv = []
    for d, sc in ds.scores.items():
        td_, pd_ = gpos[d]
        best = None
        for fk in forks:
            tf, pf = pos[fk]
            dt = tf - td_
            if abs(dt) > 5: continue
            dist = float(np.linalg.norm(pf - pd_))
            if dist > 12: continue
            key = (abs(dt), dist)
            if best is None or key < best[0]: best = (key, dt, dist, fk, 'tp' if fk in tpf else ('fp' if fk in fpf else 'ne'))
        gdiv.append(dict(t=td_, sc=int(sc), near=None if best is None else dict(dt=best[1], dist=best[2], fork=int(best[3]), lab=best[4])))
    fk_rows = [dict(fork=int(n), t=pos[n][0], lab='tp' if n in tpf else ('fp' if n in fpf else 'ne')) for n in forks]
    return dict(movie=name, set=s, gdiv=gdiv, forks=fk_rows)

if __name__ == '__main__':
    todo = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    print(len(todo), flush=True)
    with Pool(12) as p: res = p.map(job, todo, chunksize=2)
    json.dump(res, open('/workspace/cl/p21/div_timing.json', 'w'))
    from collections import Counter
    for grp, sel in (('all', lambda r: True), ('clean40', lambda r: r['set'] in ('hold36', 'prev4')), ('44b6', lambda r: r['movie'].startswith('44b6')), ('6bba', lambda r: r['movie'].startswith('6bba'))):
        R = [r for r in res if sel(r)]
        tpdt = Counter(); fnnear = Counter(); fnnone = 0; tp = fn = 0
        for r in R:
            for g in r['gdiv']:
                if g['sc']:
                    tp += 1
                    if g['near']: tpdt[g['near']['dt']] += 1
                else:
                    fn += 1
                    if g['near']: fnnear[(g['near']['dt'], g['near']['lab'])] += 1
                    else: fnnone += 1
        nf = Counter(f['lab'] for r in R for f in r['forks'])
        print(grp, 'GT div TP', tp, 'FN', fn, '| TP nearest-fork dt', dict(sorted(tpdt.items())), '| FN with a fork nearby (dt,lab)', dict(sorted(fnnear.items())), 'FN no fork nearby', fnnone, '| pred forks', dict(nf))
