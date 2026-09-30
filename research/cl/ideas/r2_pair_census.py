"""Read-only census of same-frame close pairs (rounded coords, d <= RMAX um) in P14 (= P13 + combo14 trim,long), by topology class,
with an ORACLE per pair: official-metric delta of deleting node p alone and node q alone (edge TP/FP/FN, division TP/FP).
Types: I isolated, S start (no parent, 1 child), E end (parent, no child), T through, F fork parent; suffix 'd' = fork daughter.
Also records: segment length of each node's linear segment (back+forward until start/end/fork), component size, parallel run
(consecutive frames both tracks stay within 3.5 um), distance, dz slices, dxy.
usage: python3 r2_pair_census.py [oracle 0|1]"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
for p in ['/workspace/cl', '/workspace/cl/ideas', '/workspace/official/src', '/workspace/code', '/workspace/p56stage']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np

S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
RMAX = 3.5
ORACLE = len(sys.argv) > 1 and sys.argv[1] == '1'


def job(f):
    import evalx, combo14
    from scipy.spatial import cKDTree
    name = Path(f).stem; s = f.split('ps_p13_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    nodes, edges, _ = combo14.apply(nodes, edges, order='trim,long', fullgeff=FULL[s] + '/' + name + '.geff')
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    ipos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) for n, v in nodes.items()}
    pos = {n: ipos[n] * S for n in nodes}

    def typ(n):
        c = len(out[n]); p = n in par
        t = 'F' if c >= 2 else ('T' if (p and c == 1) else ('S' if c == 1 else ('E' if p else 'I')))
        if p and len(out[par[n]]) == 2: t += 'd'
        return t

    def seg(n):
        L = 1; x = n
        while x in par and len(out[par[x]]) == 1: x = par[x]; L += 1
        x = n
        while len(out[x]) == 1: x = out[x][0]; L += 1
        return L

    adj = defaultdict(list)
    for a, bs in out.items():
        for b in bs: adj[a].append(b); adj[b].append(a)
    cid = {}; csize = []
    for n in nodes:
        if n in cid: continue
        st = [n]; cid[n] = len(csize); k = 1
        while st:
            x = st.pop()
            for y in adj.get(x, []):
                if y not in cid: cid[y] = cid[n]; st.append(y); k += 1
        csize.append(k)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    pairs = []
    for t, ns in byt.items():
        if len(ns) < 2: continue
        tr = cKDTree(np.stack([pos[n] for n in ns]))
        for i, j in tr.query_pairs(RMAX): pairs.append((ns[i], ns[j]))

    def run(a, b):
        L = 1; p, q = a, b
        while p in par and q in par:
            p, q = par[p], par[q]
            if np.linalg.norm(pos[p] - pos[q]) > 3.5: break
            L += 1
        p, q = a, b
        while len(out[p]) == 1 and len(out[q]) == 1:
            p, q = out[p][0], out[q][0]
            if np.linalg.norm(pos[p] - pos[q]) > 3.5: break
            L += 1
        return L
    rows = []
    base = evalx.score_movie(name, nodes, edges) if ORACLE else None
    KEYS = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp']
    for a, b in pairs:
        r = dict(m=name, s=s, ta=typ(a), tb=typ(b), sa=seg(a), sb=seg(b), ca=csize[cid[a]], cb=csize[cid[b]], same=cid[a] == cid[b],
                 d=float(np.linalg.norm(pos[a] - pos[b])), dz=int(abs(ipos[a][0] - ipos[b][0])),
                 dxy=float(np.linalg.norm((pos[a] - pos[b])[1:])), run=run(a, b), t=int(nodes[a]['t']))
        if ORACLE:
            for tag, x in [('a', a), ('b', b)]:
                nn = {k: v for k, v in nodes.items() if k != x}
                ne = [e for e in edges if int(e['source_id']) != x and int(e['target_id']) != x]
                rr = evalx.score_movie(name, nn, ne)
                r['o' + tag] = [rr[k] - base[k] for k in KEYS]
        rows.append(r)
    return rows


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(40, maxtasksperchild=4) as p: RR = p.map(job, fs, chunksize=1)
    rows = [r for rs in RR for r in rs]
    json.dump(rows, open('/workspace/cl/ideas/r2_pair_census%s.json' % ('_or' if ORACLE else ''), 'w'))
    print('pairs', len(rows))
    cnt = Counter()
    for r in rows:
        k = '-'.join(sorted([r['ta'], r['tb']]))
        cnt[(k, r['m'][:4])] += 1
    ks = sorted({k for k, _ in cnt}, key=lambda k: -(cnt[(k, '44b6')] + cnt[(k, '6bba')]))
    for k in ks: print('%-8s 44b6 %5d  6bba %5d' % (k, cnt[(k, '44b6')], cnt[(k, '6bba')]))
