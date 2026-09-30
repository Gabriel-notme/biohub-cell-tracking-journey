"""For each pred fork p->{c1,c2}: is there a track end s at time t near one child (alternative parent)?
Compare TP / FP / uncounted forks, on P5-style graphs."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
SETS = {'t127a': ('/workspace/t127a.txt', '/workspace/cl/p5tr/t127a', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/t127b.txt', '/workspace/cl/p5tr/t127b', '/workspace/sync3/runs/fullgraph_t127b'),
        'audit32': ('/workspace/audit32.txt', '/workspace/cl/p5tr/audit32', '/workspace/sync3/runs/fullgraph_audit32'),
        'hold36': ('/workspace/hold36.txt', '/workspace/cl/ps_p5_hold36/graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/cl/ps_p5_prev4/graphs', '/workspace/runs/fullgraph_prev4')}


def job(a):
    s, name, g, f = a
    import evalx, edge_link
    from tracking_cellmot.division_metrics import score_divisions
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(g) / (name + '.json'))
    fids, fT, fV, fE, fprob = edge_link.load_full(Path(f) / (name + '.geff'))
    fe = {(int(a_), int(b_)): float(p) for (a_, b_), p in zip(fE.tolist(), fprob.tolist())}
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tp = {inv[x] for x in res.tp_forks}; fp = {inv[x] for x in res.fp_forks}
    succ = defaultdict(list); par = {}
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id'])); par[int(e['target_id'])] = int(e['source_id'])
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    ends = defaultdict(list)
    for n, v in nodes.items():
        if not succ.get(n): ends[int(v['t'])].append(n)
    etree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in ends.items() if ns}

    def back(n, lim=30):
        k = 0
        while n in par and k < lim: n = par[n]; k += 1
        return k
    out = []
    for p, ch in succ.items():
        if len(ch) != 2: continue
        t = int(nodes[p]['t']); lab = 'TP' if p in tp else ('FP' if p in fp else 'U')
        best = None
        for c in ch:
            if t not in etree: continue
            ns, tr = etree[t]
            for j in tr.query_ball_point(pos[c], 10.0):
                sn = ns[j]
                dsc = float(np.linalg.norm(pos[c] - pos[sn])); dpc = float(np.linalg.norm(pos[c] - pos[p]))
                rec = (dsc / max(dpc, 0.1), dsc, dpc, back(sn), fe.get((sn, c), -1.), fe.get((p, c), -1.))
                if best is None or rec[0] < best[0]: best = rec
        out.append((s, lab) + ((best[0], best[1], best[2], best[3], best[4], best[5]) if best else (99, 99, 99, -1, -1, -1)))
    return out


if __name__ == '__main__':
    jobs = [(s, n, g, f) for s, (lst, g, f) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
    with Pool(40) as pool: R = [x for xs in pool.map(job, jobs) for x in xs]
    json.dump(R, open('/workspace/cl/fork_relink_probe.json', 'w'))
    for grp in [('t127a', 't127b', 'audit32'), ('hold36', 'prev4')]:
        for lab in ['TP', 'FP', 'U']:
            q = [x for x in R if x[0] in grp and x[1] == lab]
            if not q: continue
            has = [x for x in q if x[2] < 99]
            closer = [x for x in q if x[2] < 1.0]
            fe_sc = [x for x in q if x[6] >= 0]
            print('%-22s %-2s n=%5d  end within 10um of a child %4d (%.2f)  end closer than parent %4d (%.2f)  fg edge end->child %4d (%.2f)' % (
                '+'.join(grp), lab, len(q), len(has), len(has) / len(q), len(closer), len(closer) / len(q), len(fe_sc), len(fe_sc) / len(q)))
