"""GT division invariants vs predicted forks. GT: daughter distance at t+1, division time, parent history length, daughter branch
lengths. Pred (graph dir, train sets): same features for every fork, labelled TP / FP / unevaluable by the official division scorer."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])


def feats(d, ch, par, pos, t):
    kids = ch[d][:2]
    dab = float(np.linalg.norm(pos[kids[0]] - pos[kids[1]]))
    dpk = [float(np.linalg.norm(pos[k] - pos[d])) for k in kids]
    h = 0; x = d
    while x in par and len(ch[par[x]]) == 1 and h < 100: x = par[x]; h += 1
    bl = []
    for k in kids:
        L = 1; x = k
        while len(ch.get(x, [])) == 1 and L < 100: x = ch[x][0]; L += 1
        bl.append(L)
    return dict(t=t[d], dab=dab, dpk_min=min(dpk), dpk_max=max(dpk), hist=h, bmin=min(bl), bmax=max(bl), p_root=int(x not in par) if False else 0)


def gt_job(g):
    import zarr
    z = zarr.open_group(g, mode='r')
    ids = np.asarray(z['nodes/ids']).tolist(); P = np.stack([np.asarray(z['nodes/props/%s/values' % k]) for k in 'zyx'], 1) * S
    T = np.asarray(z['nodes/props/t/values']).tolist(); E = np.asarray(z['edges/ids']).tolist()
    pos = dict(zip(ids, P)); t = dict(zip(ids, T)); ch = defaultdict(list); par = {}
    for a, b in E: ch[a].append(b); par[b] = a
    return [feats(d, ch, par, pos, t) for d in ch if len(ch[d]) >= 2]


def pred_job(f):
    name = Path(f).stem
    import evalx
    from tracking_cellmot.division_metrics import score_divisions, _pred_division_fork_sets
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tp = {inv[int(x)] for x in res.tp_forks}; fp = {inv[int(x)] for x in res.fp_forks}
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}; t = {n: int(v['t']) for n, v in nodes.items()}
    out = []
    for d in [n for n in ch if len(ch[n]) >= 2]:
        r = feats(d, ch, par, pos, t); r['lab'] = 'TP' if d in tp else ('FP' if d in fp else 'U'); out.append(r)
    return out


if __name__ == '__main__':
    with Pool(64) as p:
        G = [r for rs in p.map(gt_job, sorted(glob.glob('/workspace/data/train/*.geff'))) for r in rs]
        fs = [f for T in sys.argv[1:] for f in sorted(glob.glob('/workspace/cl/ps_p8_%s/graphs/*.json' % T))]
        Pd = [r for rs in p.map(pred_job, fs) for r in rs]
    def q(v): return np.percentile(v, [0, 1, 5, 50, 95, 99, 100]).round(2)
    for k in ['dab', 'dpk_min', 'dpk_max', 't', 'hist', 'bmin', 'bmax']:
        print('%-8s GT %s | pred TP %s | pred FP %s' % (k, q([r[k] for r in G]), q([r[k] for r in Pd if r['lab'] == 'TP']), q([r[k] for r in Pd if r['lab'] == 'FP'])))
    print('pred forks', Counter(r['lab'] for r in Pd))
    # candidate rules: count TP/FP violating GT extremes
    lo = {k: min(r[k] for r in G) for k in ['dab', 'dpk_min', 'dpk_max', 't', 'hist']}; hi = {k: max(r[k] for r in G) for k in ['dab', 'dpk_min', 'dpk_max', 't']}
    print('GT ranges lo', lo, 'hi', hi)
    for k in ['dab', 'dpk_min', 'dpk_max', 't', 'hist']:
        v_lo = [r['lab'] for r in Pd if r[k] < lo[k]]
        print('  %s < GT min: %s' % (k, Counter(v_lo)))
        if k in hi:
            v_hi = [r['lab'] for r in Pd if r[k] > hi[k]]; print('  %s > GT max: %s' % (k, Counter(v_hi)))
