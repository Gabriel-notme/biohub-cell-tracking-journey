"""READ-ONLY diagnostic (critic A): anatomy of the temporal-boundary deficit in P14.
For each GT node: frame, matched?, matched distance + signed offset; if unmatched: nearest pred (dist, matched-to-other?),
the 'own-track' pred node at that frame (reached from the nearest matched GT neighbour toward the interior, following pred edges),
its distance / signed offset / match state, whether that pred track simply ends before the frame, and the nearest pre-ILP
(fullgraph) detection distance (kept or dropped). Output /workspace/cl/ideas/r3_ma_tb_rows.json."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p56stage')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def job(f):
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx
    from edge_link import load_full
    from tracking_cellmot.metrics import evaluate
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem
    st = [s for s in SETS if '_%s/' % s in f][0]
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x], float) for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gtt = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs()
    gsucc = defaultdict(list); gpar = defaultdict(list)
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(s)].append(int(d)); gpar[int(d)].append(int(s))
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    ppos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.stack([ppos[n] for n in ns]) * S), ns) for t, ns in byt.items()}
    fids, fT, fV, fE, fprob = load_full(FULL[st] + '/' + name + '.geff')
    fpos = {int(i): v for i, v in zip(fids.tolist(), fV)}
    ftrees = {}
    for t in set(fT.tolist()):
        ix = np.where(fT == t)[0]; ftrees[t] = (cKDTree(fV[ix] * S), fids[ix])
    rows = []
    for g, gp in gpos.items():
        t = gtt[g]
        r = [t]
        if g in g2p:
            p = g2p[g]; off = (ppos[p] - gp) * S
            r += [1, round(float(np.linalg.norm(off)), 2)] + [round(float(x), 2) for x in off]
            # raw (pre-ILP) position of the same id, if any
            if p in fpos:
                r += [round(float(np.linalg.norm((np.round(fpos[p]) - gp) * S)), 2)]
            else: r += [-1]
            r += [int(p not in par), int(len(succ.get(p, [])) == 0)]   # is track start / end in pred
            rows.append(r); continue
        # unmatched GT node
        r += [0]
        if t in trees:
            dd, ii = trees[t][0].query(gp * S, k=1); q = trees[t][1][ii]
            r += [round(float(dd), 2), int(q in p2g)]
        else: r += [99., 0]
        # own track: look toward interior (GT successor for early half, predecessor for late half), up to 3 steps
        own = None; how = 'nogtnb'
        direction = 'succ' if t < 50 else 'par'
        cur = g
        for k in range(1, 4):
            nxt = (gsucc if direction == 'succ' else gpar).get(cur, [])
            if len(nxt) != 1: break
            cur = nxt[0]
            if cur in g2p:
                m = g2p[cur]; pn = m; ok = True
                for _ in range(k):  # walk back toward frame t in pred
                    if direction == 'succ':
                        if pn not in par: ok = False; break
                        pn = par[pn]
                    else:
                        c = succ.get(pn, [])
                        if not c: ok = False; break
                        pn = min(c, key=lambda c_: float(np.linalg.norm((ppos[c_] - gp) * S)))
                if ok and int(nodes[pn]['t']) == t:
                    own = pn; how = 'own'
                else:
                    how = 'trk_ends'  # pred track matched nearby does not reach frame t
                break
        r += [how]
        if own is not None:
            off = (ppos[own] - gp) * S
            r += [round(float(np.linalg.norm(off)), 2)] + [round(float(x), 2) for x in off] + [int(own in p2g)]
            r += [round(float(np.linalg.norm((np.round(fpos[own]) - gp) * S)), 2) if own in fpos else -1]
        else:
            r += [-1, 0, 0, 0, 0, -1]
        if t in ftrees:
            dd, ii = ftrees[t][0].query(np.round(gp) * S, k=2)
            fid = int(ftrees[t][1][ii[0]])
            r += [round(float(dd[0]), 2), int(fid in nodes)]
        else: r += [99., 0]
        rows.append(r)
    return dict(movie=name, set=st, emb=name[:4], rows=rows)


if __name__ == '__main__':
    files = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(int(os.environ.get('RULE_POOL', '40')), maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/r3_ma_tb_rows.json', 'w'))
    print('done', len(R))
