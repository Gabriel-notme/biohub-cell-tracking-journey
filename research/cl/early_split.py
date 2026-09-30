"""'Early split' division rule (pure geometry, no learned model). A track start b at frame t (no parent) lying within D1 um of a node a
(same frame) whose parent p (frame t-1) has a as its only child, where both chains continue >= L frames and have diverged to >= D2 um
k=3 frames later, is attached as a second child of p (fork at t-1). Rationale: the detector splits a dividing nucleus into two
detections at (or one frame before) the annotated division; the official metric accepts a fork 1 frame early.
Applied post-hoc to P13 outputs; official metric per movie. usage: early_split.py '<json grid>'"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])


def apply(nodes, edges, D1=6.0, D2=8.0, L=3, K=3, dfork_k=35):
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in frames.items()}

    def fwd(n, k):
        c = [n]
        while len(c) < k + 1 and len(ch.get(c[-1], [])) == 1: c.append(ch[c[-1]][0])
        return c
    cands = []
    for b in nodes:
        if b in par: continue
        t = int(nodes[b]['t'])
        if t == 0: continue
        ns, tr = trees[t]
        for j in tr.query_ball_point(pos[b], D1):
            a = ns[j]
            if a == b or a not in par: continue
            p = par[a]
            if len(ch[p]) != 1: continue
            ca, cb = fwd(a, max(L, K)), fwd(b, max(L, K))
            if len(ca) <= L or len(cb) <= L: continue
            if len(ca) <= K or len(cb) <= K: continue
            dK = float(np.linalg.norm(pos[ca[K]] - pos[cb[K]])); d0 = float(np.linalg.norm(pos[a] - pos[b]))
            if dK < D2 or dK <= d0: continue
            cands.append((d0, p, a, b))
    cands.sort()
    used_p, used_b = set(), set(); add = []
    for d0, p, a, b in cands:
        if p in used_p or b in used_b: continue
        used_p.add(p); used_b.add(b); add.append((p, b))
    ne = list(edges) + [{'source_id': p, 'target_id': b, 'early_split': 1} for p, b in add]
    if dfork_k:
        import dfork
        ne, _ = dfork.resolve(nodes, ne, K=dfork_k)
    return ne, len(add)


def job(a):
    s, f, ci, cfg = a
    import evalx
    nodes, edges = evalx.load_graph_json(f)
    n_add = 0
    if cfg is not None: edges, n_add = apply(nodes, edges, **cfg)
    r = evalx.score_movie(Path(f).stem, nodes, edges); r['set'] = s; r['ci'] = ci; r['n_add'] = n_add; return r


if __name__ == '__main__':
    grid = json.loads(sys.argv[1])
    fs = [(s, f) for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b'] for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    cfgs = [None] + grid
    with Pool(96) as p: R = p.map(job, [(s, f, ci, c) for ci, c in enumerate(cfgs) for s, f in fs])
    json.dump({'cfgs': cfgs, 'rows': R}, open('/workspace/cl/early_split_eval.json', 'w'))
    from tracking_cellmot.metrics import summarise
    groups = [('train', lambda r: r['set'] in ('t127a', 't127b', 'audit32')), ('clean', lambda r: r['set'] in ('hold36', 'prev4')),
              ('44b6', lambda r: r['movie'].startswith('44b6')), ('6bba', lambda r: r['movie'].startswith('6bba'))]
    for ci, c in enumerate(cfgs):
        line = '%-48s' % (json.dumps(c) if c else 'P13')
        for nm, f in groups:
            rows = [r for r in R if r['ci'] == ci and f(r)]; base = [r for r in R if r['ci'] == 0 and f(r)]
            s1, s0 = summarise(rows), summarise(base)
            line += ' | %s %+.5f div %d/%d/%d add %d' % (nm, s1['score'] - s0['score'], s1['division_tp'], s1['division_fp'], s1['division_fn'], sum(r['n_add'] for r in rows))
        print(line, flush=True)
