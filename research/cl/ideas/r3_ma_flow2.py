"""READ-ONLY (critic A): (1) type of structural FN GT edges (both ends matched, not linked) by GT length and local-flow residual;
(2) precision of a GT-free flow-compensated END(t)->START(t+1) pairing: raw distance lo..14 um, residual after subtracting the local
flow (median displacement of P14 edges within 25 um) < rr um, mutual best by residual."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def job(f):
    import evalx
    from tracking_cellmot.metrics import evaluate
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x], float) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gtt = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs()
    gs = defaultdict(set); gp = defaultdict(set)
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].add(int(d)); gp[int(d)].add(int(s))
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) * S for n, v in nodes.items()}
    flow = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); flow[int(nodes[a]['t'])].append((a, pos[b] - pos[a]))
    ftr = {t: (cKDTree(np.stack([pos[a] for a, _ in L])), np.stack([d for _, d in L])) for t, L in flow.items() if L}

    def lflow(p, t):
        if t not in ftr: return np.zeros(3)
        ix = ftr[t][0].query_ball_point(p, 25.0)
        return np.median(ftr[t][1][ix], axis=0) if len(ix) >= 3 else np.zeros(3)
    # (1) structural FN typing
    typ = Counter()
    for s in gs:
        for d in gs[s]:
            ps, pd = g2p.get(s), g2p.get(d)
            if ps is None or pd is None or pd in succ.get(ps, ()): continue
            disp = gpos[d] - gpos[s]; L = float(np.linalg.norm(disp))
            res = float(np.linalg.norm(disp - lflow(gpos[s], gtt[s])))
            k = ('END' if not succ.get(ps) else 'CH') + '->' + ('START' if pd not in par else 'TAKEN')
            typ[(k, 'L>=6' if L >= 6 else 'L<6', 'res<4' if res < 4 else 'res>=4')] += 1
    # (2) GT-free flow-compensated END->START pairing
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    lab = Counter()
    for t in sorted(byt):
        ends = [n for n in byt[t] if not succ.get(n)]
        starts = [n for n in byt.get(t + 1, []) if n not in par]
        if not ends or not starts: continue
        st_tree = cKDTree(np.stack([pos[n] for n in starts]))
        cand = []
        for e in ends:
            fl = lflow(pos[e], t); pe = pos[e] + fl
            for q in st_tree.query_ball_point(pe, 4.0):
                s = starts[q]; raw = float(np.linalg.norm(pos[s] - pos[e])); res = float(np.linalg.norm(pos[s] - pe))
                cand.append((res, raw, e, s, float(np.linalg.norm(fl))))
        cand.sort()
        used_e = set(); used_s = set()
        for res, raw, e, s, fn in cand:
            if e in used_e or s in used_s: continue
            used_e.add(e); used_s.add(s)
            ge, gd = p2g.get(e), p2g.get(s)
            tp = ge is not None and gd is not None and gd in gs.get(ge, ())
            ev = (ge is not None and len(gs.get(ge, ())) > 0) or (gd is not None and len(gp.get(gd, ())) > 0)
            c = 'tp' if tp else ('fp' if ev else 'u')
            rb = 'raw<4' if raw < 4 else ('raw4-6' if raw < 6 else ('raw6-10' if raw < 10 else 'raw10+'))
            lab[(rb, 'res<2' if res < 2 else 'res2-4', 'flow>=3' if fn >= 3 else 'flow<3', c)] += 1
    return dict(emb=name[:4], typ={'|'.join(k): v for k, v in typ.items()}, lab={'|'.join(k): v for k, v in lab.items()})


if __name__ == '__main__':
    files = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(32, maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    for emb in ['44b6', '6bba']:
        T = Counter(); Lb = Counter()
        for r in R:
            if r['emb'] != emb: continue
            T.update(r['typ']); Lb.update(r['lab'])
        print('==', emb, 'structural FN types (type|len|flow-residual):')
        for k in sorted(T): print('   %-30s %d' % (k, T[k]))
        print('  flow-compensated END->START pairing (raw|res|flow): TP/FP/U')
        keys = sorted(set(k.rsplit('|', 1)[0] for k in Lb))
        for k in keys: print('   %-28s %4d / %4d / %6d' % (k, Lb[k + '|tp'], Lb[k + '|fp'], Lb[k + '|u']))
