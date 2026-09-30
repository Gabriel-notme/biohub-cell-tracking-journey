"""Per predicted edge on a set of graphs, with the OFFICIAL validity rule: an edge is evaluable if its source matches a GT node with
out_degree>0 or its target matches a GT node with in_degree>0; TP if both ends match and (gs,gt) is a GT edge, else FP.
Cutting an evaluable edge helps iff P(TP) < J/(1+J) ~ 0.48. Dump per-edge feature rows for learning."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])


def job(args):
    s, f = args
    name = Path(f).stem
    if not Path('/workspace/data/train/%s.geff/nodes/props' % name).exists(): return None
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    try: gt, _ = evalx.load_gt(name)
    except Exception: return None
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); src = ea[K.EDGE_SOURCE].to_list(); dst = ea[K.EDGE_TARGET].to_list()
    ge = set(zip(src, dst)); gout = set(src); gin = set(dst)
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    rows = []
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        valid = (a in p2g and p2g[a] in gout) or (b in p2g and p2g[b] in gin)
        if not valid: continue
        tp = int(a in p2g and b in p2g and (p2g[a], p2g[b]) in ge)
        ep = e.get('edge_prob'); ep = -1. if ep is None else float(ep)
        d = float(np.linalg.norm((np.array([nodes[a][k] for k in 'zyx']) - np.array([nodes[b][k] for k in 'zyx'])) * S))
        # local track context
        ha = 0; x = a
        while x in par and ha < 20: x = par[x]; ha += 1
        fb = 0; x = b
        while len(ch.get(x, [])) == 1 and fb < 20: x = ch[x][0]; fb += 1
        rows.append([ep, d, ha, fb, len(ch[a]), int(e.get('motion_relinked', 0) or 0), int(e.get('div_complete', 0) or 0),
                     int(a in p2g), int(b in p2g), tp])
    return s, name, rows


if __name__ == '__main__':
    SETS = json.loads(sys.argv[1]); tag = sys.argv[2]
    jobs = [(s, f) for s, d in SETS.items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(64) as p: R = [r for r in p.map(job, jobs) if r]
    np.save('/workspace/cl/edgecut_%s.npy' % tag, np.array([[hash(s) % 1000, hash(m) % 100000] + r for s, m, rs in R for r in rs], float))
    json.dump([(s, m, len(rs)) for s, m, rs in R], open('/workspace/cl/edgecut_%s_index.json' % tag, 'w'))
    for s in SETS:
        A = np.array([r for ss, m, rs in R if ss == s for r in rs])
        if not len(A): continue
        print(s, 'evaluable edges', len(A), 'TP', int(A[:, -1].sum()), 'FP', int((1 - A[:, -1]).sum()), ' FP with src unmatched', int(((A[:, -1] == 0) & (A[:, 7] == 0)).sum()), 'dst unmatched', int(((A[:, -1] == 0) & (A[:, 8] == 0)).sum()))
        for lo, hi in [(-1.5, 0), (0, .3), (.3, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.01)]:
            k = (A[:, 0] >= lo) & (A[:, 0] < hi)
            if k.sum(): print('  edge_prob [%.2f,%.2f): n %6d P(TP) %.3f' % (lo, hi, k.sum(), A[k, -1].mean()))
        for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 99)]:
            k = (A[:, 1] >= lo) & (A[:, 1] < hi)
            if k.sum(): print('  dist [%d,%d): n %6d P(TP) %.3f' % (lo, hi, k.sum(), A[k, -1].mean()))
        for nm, col in [('hist_a', 2), ('fut_b', 3)]:
            for lo, hi in [(0, 1), (1, 3), (3, 10), (10, 99)]:
                k = (A[:, col] >= lo) & (A[:, col] < hi)
                if k.sum(): print('  %s [%d,%d): n %6d P(TP) %.3f' % (nm, lo, hi, k.sum(), A[k, -1].mean()))
