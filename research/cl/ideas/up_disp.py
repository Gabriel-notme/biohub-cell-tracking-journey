"""UPSTREAM lens (read-only): displacement of P13 node coordinates from their raw pre-ILP detection (same node id), bucketed;
distance to nearest GT node (same frame) for P13 position vs raw detection position; 7um match flips (nearest-GT proxy, not the official
assignment). Also track context of large-displacement nodes (fork-adjacent, track end)."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
BK = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 99]


def job(a):
    s, f = a
    import evalx, zarr
    try: zarr.config.set({'threading.max_workers': 1})
    except Exception: pass
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    g = zarr.open_group(FULL[s] + '/' + name + '.geff', mode='r')
    fid = np.asarray(g['nodes/ids'][:]).astype(np.int64)
    fP = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'tzyx'], 1)
    fidx = {int(i): j for j, i in enumerate(fid.tolist())}
    gt, n_total = evalx.load_gt(name)
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gT = np.array(ga['t'].to_list()); gX = np.stack([np.array(ga[k].to_list()) for k in 'zyx'], 1) * S
    trees = {int(t): cKDTree(gX[gT == t]) for t in np.unique(gT)}
    ch = defaultdict(int); par = set()
    for e in edges: ch[int(e['source_id'])] += 1; par.add(int(e['target_id']))
    C = Counter(); D = defaultdict(list)
    for n, v in nodes.items():
        j = fidx.get(n)
        if j is None or int(fP[j, 0]) != int(v['t']): continue
        # evaluation rounds coordinates to int voxels
        p = np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) * S
        r = np.array([max(0, int(round(x))) for x in fP[j, 1:]], float) * S
        disp = float(np.linalg.norm(p - r))
        b = int(np.searchsorted(BK, disp, side='right') - 1)
        ctx = 'fork' if ch[n] >= 2 else ('end' if ch[n] == 0 or n not in par else 'mid')
        C[(b, ctx, 'n')] += 1
        t = int(v['t'])
        if t not in trees: continue
        dp, _ = trees[t].query(p); dr, _ = trees[t].query(r)
        if min(dp, dr) > 10: continue
        C[(b, ctx, 'nearGT')] += 1
        C[(b, ctx, 'p_in_r_out')] += int(dp <= 7 and dr > 7)
        C[(b, ctx, 'r_in_p_out')] += int(dr <= 7 and dp > 7)
        C[(b, ctx, 'raw_closer')] += int(dr < dp)
        D[(b, ctx)].append(dr - dp)
    return {'set': s, 'movie': name, 'C': {json.dumps(k): v for k, v in C.items()}, 'D': {json.dumps(k): [float(np.sum(v)), len(v)] for k, v in D.items()}}


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(32) as p: R = p.map(job, jobs, chunksize=1)
    T = defaultdict(Counter); DD = defaultdict(lambda: np.zeros(2))
    for r in R:
        e = r['movie'][:4]
        for k, v in r['C'].items(): b, ctx, m = json.loads(k); T[(e, b, ctx)][m] += v
        for k, v in r['D'].items(): b, ctx = json.loads(k); DD[(e, b, ctx)] += np.array(v)
    for e in ['44b6', '6bba']:
        for ctx in ['mid', 'end', 'fork']:
            for b in range(len(BK) - 1):
                c = T[(e, b, ctx)]
                if not c['n']: continue
                sm, nn = DD[(e, b, ctx)]
                print('%s %-4s disp [%.1f,%.1f) n %7d nearGT %6d | raw closer %.3f mean(d_raw-d_p13) %+.3f um | p13-only-in7 %4d raw-only-in7 %4d' % (
                    e, ctx, BK[b], BK[b + 1], c['n'], c['nearGT'], c['raw_closer'] / max(1, c['nearGT']), sm / max(1, nn), c['p_in_r_out'], c['r_in_p_out']))
