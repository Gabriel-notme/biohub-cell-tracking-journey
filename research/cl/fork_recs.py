"""Per-fork records (TP / FP / uncounted) with structural features, for a set of graphs."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import evalx
from tracking_cellmot.division_metrics import score_divisions
S = np.array([1.625, .40625, .40625])

def feats(nodes, edges):
    succ = defaultdict(list); par = {}
    flag = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
        flag[(s, d)] = 'dc' if e.get('div_complete') else ('dsr' if e.get('dsr') else '')
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    def fwd(n, lim=40):
        k = 0
        while len(succ.get(n, [])) == 1 and k < lim: n = succ[n][0]; k += 1
        return k, len(succ.get(n, []))
    def back(n, lim=40):
        k = 0
        while n in par and k < lim: n = par[n]; k += 1
        return k
    out = {}
    for p, ch in succ.items():
        if len(ch) != 2: continue
        a, b = ch
        la, ea = fwd(a); lb, eb = fwd(b)
        out[p] = dict(t=int(nodes[p]['t']), hist=back(p), la=la, lb=lb, ea=ea, eb=eb,
                      d_pa=float(np.linalg.norm(pos[a] - pos[p])), d_pb=float(np.linalg.norm(pos[b] - pos[p])), d_ab=float(np.linalg.norm(pos[a] - pos[b])),
                      src='|'.join(sorted(x for x in (flag[(p, a)], flag[(p, b)]) if x)) or 'base')
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
    return recs, sum(res.scores.values()), len(res.scores)

if __name__ == '__main__':
    lst, gdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    names = [l.strip() for l in open(lst) if l.strip()]
    with Pool(min(32, len(names))) as pool:
        R = pool.map(job, [(n, gdir) for n in names])
    recs = [r for rs, _, _ in R for r in rs]
    json.dump(recs, open(out, 'w'))
    from collections import Counter
    print('GT divs', sum(x[2] for x in R), 'TP', sum(x[1] for x in R), Counter(r['lab'] for r in recs), Counter((r['lab'], r['src']) for r in recs if r['lab'] != 'U'))
