"""Read-only analysis of P13 forks: TP timing offsets vs GT divider, FP categories (cross-component / evaluable no-div /
near-GT-div timing or duplicate), fork origin, daughter geometry. usage: python3 ideas/pp_fork_an.py"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np

SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
S = np.array([1.625, .40625, .40625])


def job(f):
    import evalx
    import tracking_cellmot.division_metrics as DM
    K = evalx.K
    name = Path(f).stem; st = f.split('ps_p13_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    succ = defaultdict(list); par = {}; eo = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
        eo[(s, d)] = 'dc' if 'div_complete' in e else ('rl' if 'relink' in e else ('el' if 'edge_link' in e else 'b5'))
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    sd = DM.score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    ev, cross, malf = DM._pred_division_fork_sets(pred, gt, evalx.SCALE, 7.)
    tp = {inv[x] for x in sd.tp_forks}; fpf = {inv[x] for x in sd.fp_forks}
    ev = {inv[x] for x in ev}; cross = {inv[x] for x in cross}
    mp = DM._match_full(pred, gt, evalx.SCALE, 7.)
    ma = DM._matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gt_t = dict(zip(ga[K.NODE_ID].to_list(), ga['t'].to_list()))
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gdiv = [n for n in gt.node_ids() if gt.out_degree(n) >= 2]
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    recs = []
    for f0 in [n for n in nodes if len(succ.get(n, [])) >= 2]:
        lab = 'TP' if f0 in tp else ('FP' if f0 in fpf else 'unl')
        if lab == 'unl': continue
        t0 = int(nodes[f0]['t'])
        # nearest GT division in space-time
        best = None
        for g in gdiv:
            dt = t0 - int(gt_t[g]); d = float(np.linalg.norm(gpos[g] - pos[f0]))
            if abs(dt) <= 4 and d <= 12:
                k = (abs(dt), d)
                if best is None or k < best[0]: best = (k, dt, d)
        a, b = succ[f0][:2]
        r = dict(movie=name, set=st, f=f0, t=t0, lab=lab, cross=int(f0 in cross), ev=int(f0 in ev), matched=int(f0 in p2g),
                 orig='+'.join(sorted({eo[(f0, x)] for x in succ[f0]})), near_dt=None if best is None else best[1], near_d=None if best is None else round(best[2], 2),
                 d_ab=round(float(np.linalg.norm(pos[a] - pos[b])), 2), d_pa=round(float(np.linalg.norm(pos[a] - pos[f0])), 2), d_pb=round(float(np.linalg.norm(pos[b] - pos[f0])), 2),
                 kids_matched=int(a in p2g) + int(b in p2g))
        recs.append(r)
    return recs


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(48) as p: R = p.map(job, fs, chunksize=1)
    recs = [r for rs in R for r in rs]
    json.dump(recs, open('/workspace/cl/ideas/pp_fork_recs.json', 'w'))
    print('n', len(recs), Counter(r['lab'] for r in recs))
    print('TP near_dt', Counter((r['near_dt'], r['movie'][:4]) for r in recs if r['lab'] == 'TP'))
    print('FP near_dt', Counter((r['near_dt'], r['movie'][:4]) for r in recs if r['lab'] == 'FP'))
    print('FP cross/ev', Counter((r['cross'], r['ev'], r['orig'], r['kids_matched']) for r in recs if r['lab'] == 'FP'))
    for r in recs:
        if r['lab'] == 'FP': print(r)
