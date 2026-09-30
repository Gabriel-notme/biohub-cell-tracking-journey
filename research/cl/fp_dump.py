"""Dump every evaluable predicted fork (TP / FP) of a config's outputs with provenance and structure, to look for FP patterns."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])


def job(a):
    s, f = a
    name = Path(f).stem
    import evalx
    from tracking_cellmot.division_metrics import score_divisions
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tp = {inv[int(x)] for x in res.tp_forks}; fp = {inv[int(x)] for x in res.fp_forks}
    ch = defaultdict(list); par = {}; eat = {}
    for e in edges:
        a_, b_ = int(e['source_id']), int(e['target_id']); ch[a_].append(b_); par[b_] = a_; eat[(a_, b_)] = e
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    out = []
    for d in tp | fp:
        if len(ch[d]) < 2: continue
        kids = ch[d][:2]
        prov = ['dc' if eat[(d, k)].get('div_complete') else ('el' if eat[(d, k)].get('edge_link') else 'b5') for k in kids]
        h = 0; x = d
        while x in par and len(ch[par[x]]) == 1 and h < 100: x = par[x]; h += 1
        bl = []
        for k in kids:
            L = 1; y = k
            while len(ch.get(y, [])) == 1 and L < 100: y = ch[y][0]; L += 1
            bl.append((L, len(ch.get(y, []))))
        ep = [eat[(d, k)].get('edge_prob') for k in kids]
        out.append(dict(set=s, movie=name, t=int(nodes[d]['t']), lab='TP' if d in tp else 'FP', prov='+'.join(sorted(prov)), hist=h,
                        bl=sorted(bl), dab=float(np.linalg.norm(pos[kids[0]] - pos[kids[1]])), ep=ep, root_t=int(nodes[x]['t']), root_has_par=x in par))
    return out


if __name__ == '__main__':
    cfg = sys.argv[1]
    jobs = [(s, f) for s in sys.argv[2:] for f in sorted(glob.glob('/workspace/cl/ps_%s_%s/graphs/*.json' % (cfg, s)))]
    with Pool(64) as p: R = [r for rs in p.map(job, jobs) for r in rs]
    json.dump(R, open('/workspace/cl/fp_dump_%s.json' % cfg, 'w'))
    print(Counter((r['lab'], r['prov']) for r in R))
    for r in sorted(R, key=lambda r: (r['lab'], r['prov'])):
        if r['lab'] == 'FP': print(r)
    for lab in ['TP', 'FP']:
        Q = [r for r in R if r['lab'] == lab]
        print(lab, 'hist q', np.percentile([r['hist'] for r in Q], [10, 50, 90]), 'min branch q', np.percentile([r['bl'][0][0] for r in Q], [10, 50, 90]),
              'dab q', np.percentile([r['dab'] for r in Q], [10, 50, 90]).round(1), 'branch ends in fork', sum(1 for r in Q if any(b[1] >= 2 for b in r['bl'])))
