"""Stolen-type division: does the stealer q have an alternative continuation at t+1? usage: alt_cont.py <labeled.json> <graph_dir> <full_dir> <out.json>"""
import os, sys, json
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds')
import numpy as np
S = np.array([1.625, .40625, .40625], np.float32)
lab_path, gdir, fdir, outp = sys.argv[1:5]
def job(args):
    name, cands = args
    import evalx, dsr
    from scipy.spatial import cKDTree
    nodes, edges = evalx.load_graph_json(gdir + '/' + name + '.json')
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * S for n, v in nodes.items()}
    frames = defaultdict(list); starts = defaultdict(list)
    for n, v in nodes.items():
        frames[int(v['t'])].append(n)
        if n not in prev: starts[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}
    stree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in starts.items() if ns}
    FT, FV = dsr.load_full(fdir + '/' + name + '.geff'); FP = FV * S
    dropped = {}
    for t in np.unique(FT):
        idx = np.where(FT == t)[0]
        if int(t) in trees:
            d, _ = trees[int(t)].query(FP[idx]); idx = idx[d > 4.0]
        if len(idx): dropped[int(t)] = cKDTree(FP[idx])
    res = []
    for c in cands:
        q, b, p = c['q'], c['b'], c['p']
        t = int(nodes[q]['t'])
        v = pos[q] - pos[prev[q]] if q in prev else np.zeros(3, np.float32)
        pr = pos[q] + v
        r = dict(c)
        # nearest start at t+1 other than b
        ds = 99.; 
        if t + 1 in stree:
            ns, tr = stree[t + 1]
            dd, jj = tr.query(pr, k=3)
            for d_, j_ in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
                if j_ < len(ns) and ns[j_] != b: ds = float(d_); break
        dd_ = 99.
        if t + 1 in dropped:
            dd_, _ = dropped[t + 1].query(pr); dd_ = float(dd_)
        r['alt_start'] = ds; r['alt_drop'] = dd_
        r['d_qb'] = float(np.linalg.norm(pos[q] - pos[b])); r['d_pred_b'] = float(np.linalg.norm(pr - pos[b]))
        r['q_hist'] = 0; x = q
        while x in prev and r['q_hist'] < 20: x = prev[x]; r['q_hist'] += 1
        r['b_fut'] = 0; x = b
        while len(out.get(x, [])) == 1 and r['b_fut'] < 20: x = out[x][0]; r['b_fut'] += 1
        res.append(r)
    return res
if __name__ == '__main__':
    L = json.load(open(lab_path))
    by = defaultdict(list)
    for c in L:
        if c['typ'] == 'stolen' and c['lab'] in ('pos', 'neg'): by[c['movie']].append(c)
    del L
    with Pool(24) as pool: rr = pool.map(job, list(by.items()))
    rows = [r for x in rr for r in x]
    json.dump(rows, open(outp, 'w'))
    P = [r for r in rows if r['lab'] == 'pos']; N = [r for r in rows if r['lab'] == 'neg']
    print('pos', len(P), 'neg', len(N))
    for k, th in [('alt_start', 4), ('alt_start', 6), ('alt_drop', 3), ('alt_drop', 5)]:
        fp_ = np.mean([r[k] <= th for r in P]); fn_ = np.mean([r[k] <= th for r in N])
        print('%s<=%g  pos %.3f  neg %.4f' % (k, th, fp_, fn_))
    for r in sorted(P, key=lambda r: -r['fork']):
        print('POS %s p%d q%d fork %.4f alt_start %.1f alt_drop %.1f d_qb %.1f d_pb %.1f d_pred_b %.1f e_old %.3f e_pb %.3f qh %d bf %d' % (r['movie'], r['p'], r['q'], r['fork'], r['alt_start'], r['alt_drop'], r['d_qb'], r['d_pb'], r['d_pred_b'], r['e_old'] or -1, r['e_pb'], r['q_hist'], r['b_fut']))
    # among negatives with fork>=0.5
    H = [r for r in N if r['fork'] >= 0.5]
    print('neg fork>=0.5', len(H), 'alt_start<=6', sum(r['alt_start'] <= 6 for r in H), 'alt_drop<=5', sum(r['alt_drop'] <= 5 for r in H))
