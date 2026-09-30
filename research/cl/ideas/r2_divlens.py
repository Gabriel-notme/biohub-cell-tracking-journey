"""Read-only division-structure lens on P14 graphs (P13 + combo14 when the P14 file is not there yet).
Per fork: official label (score_divisions tp/fp + evaluable/cross sets), origin, branch lengths/ends, parent history,
daughter collision with other tracks, daughter-daughter parallelism, proximity to nodes removed by term_trim.
Writes /workspace/cl/ideas/r2_divlens_rows.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
FIN = set()


def rpos(v):
    return np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S


def job(a):
    s, f13, use14 = a
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx
    from scipy.spatial import cKDTree
    from tracking_cellmot.division_metrics import score_divisions, _pred_division_fork_sets
    name = Path(f13).stem
    n13, e13 = evalx.load_graph_json(f13)
    f14 = '/workspace/cl/ps_p14_%s/graphs/%s.json' % (s, name)
    if use14 and os.path.exists(f14):
        nodes, edges = evalx.load_graph_json(f14)
    else:
        from ideas import combo14
        nodes, edges, _ = combo14.apply(n13, e13, fullgeff=FULL[s] + '/' + name + '.geff')
    bn, be = evalx.load_graph_json(B5[s] + '/' + name + '.json')
    bch = defaultdict(list)
    for e in be: bch[int(e['source_id'])].append(int(e['target_id']))
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    ev, cross, mal = _pred_division_fork_sets(pred, gt, evalx.SCALE, 7.)
    tp = {inv[int(x)] for x in res.tp_forks}; fp = {inv[int(x)] for x in res.fp_forks}
    ev = {inv[int(x)] for x in ev}; cross = {inv[int(x)] for x in cross}
    ch = defaultdict(list); par = {}; eattr = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u; eattr[(u, v)] = e
    pos = {n: rpos(v) for n, v in nodes.items()}; t = {n: int(v['t']) for n, v in nodes.items()}
    Tmax = max(t.values())
    byt = defaultdict(list)
    for n in nodes: byt[t[n]].append(n)
    trees = {tt: (cKDTree(np.stack([pos[n] for n in ns])), ns) for tt, ns in byt.items()}
    # nodes removed by term_trim (in P13 but not in P14)
    removed = [n for n in n13 if n not in nodes]
    rem_by_t = defaultdict(list)
    for n in removed: rem_by_t[int(n13[n]['t'])].append(rpos(n13[n]))
    e13s = {(int(e['source_id']), int(e['target_id'])) for e in e13}
    added = [(int(e['source_id']), int(e['target_id'])) for e in edges if (int(e['source_id']), int(e['target_id'])) not in e13s]
    added_nodes = {x for ab in added for x in ab}
    ch13 = defaultdict(list)
    for u, v in e13s: ch13[u].append(v)

    def branch(k, maxl=200):
        L = [k]
        while len(ch.get(L[-1], [])) == 1 and len(L) < maxl: L.append(ch[L[-1]][0])
        return L, len(ch.get(L[-1], []))

    def hist(x):
        h = 0
        while x in par and len(ch[par[x]]) == 1: x = par[x]; h += 1
        return h, (x in par)

    rows = []
    forks = [n for n in ch if len(ch[n]) >= 2]
    for p in forks:
        kids = ch[p][:2]
        br = [branch(k) for k in kids]
        own = {p} | set(br[0][0]) | set(br[1][0])
        hh = hist(p)
        # collision: first 3 nodes of each daughter branch vs other nodes of the same frame (not own lineage)
        coll = []
        for bi, (L, _) in enumerate(br):
            cm = []
            for x in L[:3]:
                tr, ns = trees[t[x]]
                dd, ii = tr.query(pos[x], k=min(6, len(ns)))
                dd = np.atleast_1d(dd); ii = np.atleast_1d(ii)
                best = 99.
                for d_, i_ in zip(dd, ii):
                    if i_ < len(ns) and ns[i_] not in own: best = float(d_); break
                cm.append(round(best, 2))
            coll.append(cm)
        # parallel daughters: distance between branch nodes at equal offsets 0..5
        dab = [round(float(np.linalg.norm(pos[br[0][0][i]] - pos[br[1][0][i]])), 2) for i in range(min(6, len(br[0][0]), len(br[1][0])))]
        # parent-side collision (p and its 2 predecessors) with non-own nodes
        # term_trim proximity: removed P13 nodes within 6um of lineage nodes t-2..t+5
        tn = 99.
        for x in [p] + br[0][0][:5] + br[1][0][:5] + ([par[p]] if p in par else []):
            for q in rem_by_t.get(t[x], []):
                tn = min(tn, float(np.linalg.norm(q - pos[x])))
        touched = bool(own & added_nodes)
        was13 = len(ch13.get(p, [])) >= 2 and sorted(ch13[p][:2]) == sorted(kids)
        dc = any('div_complete' in eattr[(p, k)] for k in kids)
        origin = 'dc' if dc else ('b5' if sorted(bch.get(p, [])) == sorted(kids) else 'mod')
        lab = 'TP' if p in tp else ('FP' if p in fp else 'U')
        rows.append(dict(set=s, movie=name, p=p, t=t[p], Tmax=Tmax, lab=lab, ev=p in ev, cross=p in cross, origin=origin,
                         L=[len(br[0][0]), len(br[1][0])], Lend=[br[0][1], br[1][1]], tend=[t[br[0][0][-1]], t[br[1][0][-1]]],
                         hist=hh[0], hist_fork=hh[1], coll=coll, dab=dab, trim_near=round(tn, 2), touched=touched, was13=was13,
                         dpa=[round(float(np.linalg.norm(pos[k] - pos[p])), 2) for k in kids]))
    return rows


if __name__ == '__main__':
    log = open('/workspace/cl/p14all.log').read()
    fin = {x for x in SETS if ('FIN %s ' % x) in log}
    jobs = [(s, f, s in fin) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(32, maxtasksperchild=4) as p: R = [r for rs in p.map(job, jobs, chunksize=1) for r in rs]
    json.dump(R, open('/workspace/cl/ideas/r2_divlens_rows.json', 'w'))
    from collections import Counter
    print('forks', len(R), Counter(r['lab'] for r in R), 'fin', sorted(fin))
