"""Per-fork records (TP / FP / uncounted) with richer structural features."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
import evalx
from tracking_cellmot.division_metrics import score_divisions
S = np.array([1.625, .40625, .40625])

def feats(nodes, edges):
    succ = defaultdict(list); par = {}; flag = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
        flag[(s, d)] = 'dc' if e.get('div_complete') else ('dsr' if e.get('dsr') else '')
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in frames.items()}
    forks = [p for p, ch in succ.items() if len(ch) == 2]
    fpos = np.array([np.r_[nodes[p]['t'] * 3.0, pos[p]] for p in forks]) if forks else np.zeros((0, 4))
    ftree = cKDTree(fpos) if forks else None
    def fwd(n, lim=40):
        k = 0
        while len(succ.get(n, [])) == 1 and k < lim: n = succ[n][0]; k += 1
        return k, len(succ.get(n, []))
    def back(n, lim=40):
        k = 0
        while n in par and k < lim: n = par[n]; k += 1
        return k
    def dens(t, x, r):
        if t not in trees: return 0
        return len(trees[t][1].query_ball_point(x, r))
    out = {}
    for p in forks:
        a, b = succ[p]
        la, ea = fwd(a); lb, eb = fwd(b)
        if (la, lb) > (lb, la): pass
        va, vb = pos[a] - pos[p], pos[b] - pos[p]
        cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6))
        pp = par.get(p); vel = float(np.linalg.norm(pos[p] - pos[pp])) if pp is not None else -1.0
        t = int(nodes[p]['t'])
        mid = (pos[a] + pos[b]) / 2
        # nearest other node distances
        ns, tr = trees[t]; dd, _ = tr.query(pos[p], k=min(2, len(ns))); nn_p = float(np.atleast_1d(dd)[-1]) if len(ns) > 1 else 99.
        nf = len(ftree.query_ball_point(np.r_[t * 3.0, pos[p]], 15.0)) - 1
        src = '|'.join(sorted(x for x in (flag[(p, a)], flag[(p, b)]) if x)) or 'base'
        out[p] = dict(t=t, z=float(pos[p][0]), hist=back(p), la=min(la, lb), lb=max(la, lb), ea=ea, eb=eb, anyfork=int(ea == 2 or eb == 2),
                      d_near=float(min(np.linalg.norm(va), np.linalg.norm(vb))), d_far=float(max(np.linalg.norm(va), np.linalg.norm(vb))),
                      d_ab=float(np.linalg.norm(pos[a] - pos[b])), cos=cos, dz_ab=float(abs(pos[a][0] - pos[b][0])), vel=vel,
                      dens_p=dens(t, pos[p], 8.0), dens_mid=dens(t + 1, mid, 8.0), nn_p=nn_p, nfork=nf,
                      src_dc=int('dc' in src), src_dsr=int('dsr' in src), src=src)
    return out

def job(args):
    name, gdir = args
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    F = feats(nodes, edges)
    tp = {inv[x] for x in res.tp_forks}; fp = {inv[x] for x in res.fp_forks}
    recs = []
    for p, f in F.items():
        f.update(movie=name, node=p, lab='TP' if p in tp else ('FP' if p in fp else 'U')); recs.append(f)
    return recs

if __name__ == '__main__':
    sets = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/p3_hold36/graphs'), 'prev4': ('/workspace/preview4.txt', '/workspace/runs/p3_prev4/graphs'),
            'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/p3_audit32/graphs'), 't127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/fin_p3_t127a/graphs'),
            't127b': ('/workspace/t127b.txt', '/workspace/sync3/runs/fin_p3_t127b/graphs')}
    jobs = []
    for s, (lst, g) in sets.items():
        for n in [l.strip() for l in open(lst) if l.strip()]: jobs.append((s, n, g))
    with Pool(48) as pool:
        R = pool.map(job, [(n, g) for s, n, g in jobs])
    allr = []
    for (s, n, g), rs in zip(jobs, R):
        for r in rs: r['set'] = s; allr.append(r)
    json.dump(allr, open('/workspace/cl/forks2_p3.json', 'w'))
    from collections import Counter
    print(Counter((r['set'], r['lab']) for r in allr))
